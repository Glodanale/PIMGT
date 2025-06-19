import cudf
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh

# ---------- STEP 1: Load CSV ----------
def load_cleaned_location(filepath):
    return cudf.read_csv(filepath)

# ---------- STEP 2: Define Segments ----------
def define_segments_cudf(df, segment_size=50.0):
    min_x = df["Local_X_m"].min()
    min_y = df["Local_Y_m"].min()

    df["x_bin"] = ((df["Local_X_m"] - min_x) // segment_size).astype("int32")
    df["y_bin"] = ((df["Local_Y_m"] - min_y) // segment_size).astype("int32")

    df["segment_id"] = df["Lane_ID"].astype(str) + "_" + df["x_bin"].astype(str) + "_" + df["y_bin"].astype(str)

    segment_ids = df["segment_id"].unique().to_pandas().sort_values()

    # Optional: you can build a segment lookup DataFrame if needed
    return df


# ---------- STEP 3: Detect Gaps ----------
def detect_time_segments(df, gap_threshold=500):
    times = df["timestamp"].drop_duplicates().sort_values().to_pandas().values
    gaps = np.diff(times)
    starts = [times[0]] + [times[i + 1] for i in range(len(gaps)) if gaps[i] > gap_threshold]
    ends = [times[i] for i in range(len(gaps)) if gaps[i] > gap_threshold] + [times[-1]]
    return list(zip(starts, ends))

# ---------- STEP 4: Aggregate Node Features ----------
def aggregate_node_features_cudf(df, time_col="Global_Time"):
    """
    Aggregates node features (vehicle count, mean speed, variance of speed, density, and flow)
    for each segment_id (defined by Lane_ID + 2D spatial bin) at each timestamp.

    Args:
        df (cudf.DataFrame): Input dataframe with 'segment_id' assigned.
        time_col (str): Column name for time tracking (typically 'Global_Time').

    Returns:
        cudf.DataFrame: Aggregated node features per segment_id per timestamp.
    """

    df = df.dropna(subset=["segment_id"])
    df["segment_id"] = df["segment_id"].astype("str")  # Ensure it is string type
    time_steps = df[time_col].unique().to_pandas().sort_values()

    results = []

    for t in tqdm(time_steps, desc="Aggregating timestamps", unit="step"):
        df_t = df[df[time_col] == t]

        # Group by the new 2D-based segment_id
        grouped = df_t.groupby("segment_id").agg({
            "Vehicle_ID": "count",
            "v_Vel_mps": ["mean", "var"]
        }).reset_index()

        grouped.columns = ["segment_id", "vehicle_count", "mean_speed", "var_speed"]

        # Assume area (50x50m) per grid cell
        grid_area = 50.0 * 50.0  # meters squared
        grid_width = 50.0  # 50 meters in one direction for density calculation
        
        grouped["density"] = grouped["vehicle_count"] / grid_width  # vehicles per 50m strip
        grouped["flow"] = grouped["density"] * grouped["mean_speed"]
        grouped["timestamp"] = t

        results.append(grouped)

    return cudf.concat(results)


# ---------- STEP 5: Generate Sequences ----------
def generate_sequences_by_segments(df, in_len=30, out_len=10, stride=10):
    """
    Generates sequences for model training from the 2D spatial + lane segmented node features.

    Args:
        df (cudf.DataFrame): Aggregated node features dataframe (output from aggregate_node_features_cudf).
        in_len (int): Number of input steps.
        out_len (int): Number of output steps.
        stride (int): Stride between samples.

    Returns:
        dict: {'x': input sequences, 'y': output sequences, 'xtime': input times, 'ytime': output times}
    """
    segments = detect_time_segments(df)
    node_ids = df["segment_id"].unique().to_pandas().sort_values().tolist()
    num_nodes = len(node_ids)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    feature_cols = ["density", "mean_speed", "flow"]

    x_list, y_list, xtime, ytime = [], [], [], []

    for seg_start, seg_end in segments:
        seg_df = df[(df["timestamp"] >= seg_start) & (df["timestamp"] <= seg_end)]
        seg_timestamps = seg_df["timestamp"].unique().to_pandas().sort_values().tolist()

        iterator = tqdm(
            range(0, len(seg_timestamps) - in_len - out_len + 1, stride),
            desc=f"Segment {seg_start} → {seg_end}",
            unit="samples"
        )

        for i in iterator:
            in_times = seg_timestamps[i:i + in_len]
            out_times = seg_timestamps[i + in_len:i + in_len + out_len]

            x_seq = np.zeros((in_len, num_nodes, len(feature_cols)), dtype=np.float32)
            y_seq = np.zeros((out_len, num_nodes, len(feature_cols)), dtype=np.float32)

            for j, t in enumerate(in_times):
                df_t = seg_df[seg_df["timestamp"] == t].to_pandas()
                for _, row in df_t.iterrows():
                    if row["segment_id"] in node_index:
                        x_seq[j, node_index[row["segment_id"]]] = [row[f] for f in feature_cols]

            for j, t in enumerate(out_times):
                df_t = seg_df[seg_df["timestamp"] == t].to_pandas()
                for _, row in df_t.iterrows():
                    if row["segment_id"] in node_index:
                        y_seq[j, node_index[row["segment_id"]]] = [row[f] for f in feature_cols]

            x_list.append(x_seq)
            y_list.append(y_seq)
            xtime.append(in_times)
            ytime.append(out_times)

    return {
        "x": np.array(x_list),
        "y": np.array(y_list),
        "xtime": np.array(xtime),
        "ytime": np.array(ytime)
    }


# ---------- STEP 6: Train/Val/Test Split ----------
def split_and_save_dataset(data, base_path):
    total = len(data["x"])
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    splits = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, total)
    }

    os.makedirs(base_path, exist_ok=True)

    for name, s in splits.items():
        split_data = {k: v[s] for k, v in data.items()}
        with open(os.path.join(base_path, f"{name}.pkl"), "wb") as f:
            pickle.dump(split_data, f)

    print(f"\nSaved dataset splits to: {base_path}")
    return {k: len(data['x'][s]) for k, s in splits.items()}


def build_graph_conn(df, node_index, segment_size=50.0):
    """
    Builds the physical connection graph (graph_conn) with:
    - Same lane → adjacent grid
    - Same bin → adjacent lanes
    - Adjacent bins → adjacent lanes
    """

    print("Building updated graph_conn adjacency matrix with lane neighbor support...")

    # Convert to pandas for easier processing
    df_pd = df[["segment_id", "Lane_ID", "x_bin", "y_bin"]].drop_duplicates().to_pandas()

    # Map segment_id to (Lane_ID, x_bin, y_bin)
    segment_lookup = {}
    for _, row in df_pd.iterrows():
        segment_lookup[row["segment_id"]] = (row["Lane_ID"], row["x_bin"], row["y_bin"])

    num_nodes = len(node_index)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for seg_id_i, idx_i in node_index.items():
        lane_i, x_i, y_i = segment_lookup[seg_id_i]

        for seg_id_j, idx_j in node_index.items():
            if seg_id_i == seg_id_j:
                adjacency[idx_i, idx_j] = 1.0  # Always connect self-loop
                continue

            lane_j, x_j, y_j = segment_lookup[seg_id_j]

            x_diff = abs(x_i - x_j)
            y_diff = abs(y_i - y_j)
            lane_diff = abs(lane_i - lane_j)

            # Case 1: Same Lane, Adjacent Grid
            if lane_i == lane_j and (x_diff <= 1 and y_diff <= 1):
                adjacency[idx_i, idx_j] = 1.0

            # Case 2: Same Grid, Adjacent Lanes
            elif x_diff == 0 and y_diff == 0 and lane_diff == 1:
                adjacency[idx_i, idx_j] = 1.0

            # Case 3: Adjacent Grid, Adjacent Lanes
            elif (x_diff <= 1 and y_diff <= 1) and lane_diff == 1:
                adjacency[idx_i, idx_j] = 1.0

    print(f"Updated graph_conn built: {adjacency.shape} matrix with {int(adjacency.sum())} edges (including self-loops)")
    return adjacency



def build_graph_sml(complete_time_series, similarity_delta=0.1):
    """
    Builds the feature-based similarity graph (graph_sml) for the NGSIM dataset,
    following the custom similarity used in the original MGT utils.py.

    Args:
        complete_time_series (np.ndarray): Full time series (time_steps, num_nodes, num_features).
        similarity_delta (float): Similarity threshold for retaining edges.

    Returns:
        np.ndarray: Symmetric graph (num_nodes, num_nodes), with self-loops.
    """

    print("Building graph_sml with custom similarity measure...")

    # (time_steps, num_nodes, num_features) -> (num_nodes, flattened)
    time_steps, num_nodes, num_features = complete_time_series.shape
    feature_traces = complete_time_series.transpose(1, 0, 2).reshape(num_nodes, -1)

    n = feature_traces.shape[0]
    graph_sml = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i, n):
            a = np.linalg.norm(feature_traces[i] - feature_traces[j])**2
            b = np.minimum(np.linalg.norm(feature_traces[i])**2, np.linalg.norm(feature_traces[j])**2)
            if b == 0:  # avoid divide by zero
                sim = 0.0
            else:
                sim = np.exp(-a / b)

            if sim > similarity_delta:
                graph_sml[i, j] = graph_sml[j, i] = sim

    # Force self-loops
    np.fill_diagonal(graph_sml, 1.0)

    print(f"graph_sml built: {graph_sml.shape} matrix with {int(graph_sml.sum())} total edges (including self-loops)")
    return graph_sml



def build_eigenmaps(graph_conn, k=8):
    """
    Computes eigenmaps (spectral embeddings) from the physical connection graph.

    Args:
        graph_conn (np.ndarray): Adjacency matrix (num_nodes, num_nodes), with self-loops.
        k (int): Number of eigenvectors to keep.

    Returns:
        np.ndarray: Eigenmaps (num_nodes, k)
    """

    print("Computing eigenmaps from graph_conn...")

    A = graph_conn.copy()

    # 1. Symmetrize and zero out diagonals (no self loops)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)

    # 2. Check if graph is connected
    n_components = connected_components(csr_matrix(A), directed=False, return_labels=False)
    assert n_components == 1, "Graph is not connected!"

    n = A.shape[0]

    # 3. Build normalized Laplacian
    degree = np.sum(A, axis=1)
    degree[degree == 0] = 1e-8  # Avoid division by zero
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # 4. Solve eigen decomposition
    _, eigenvectors = eigh(L)

    # 5. Keep first k non-trivial eigenvectors (skip trivial constant eigenvector)
    eigenmaps = eigenvectors[:, 1:k+1]

    print(f"Eigenmaps computed: {eigenmaps.shape} (num_nodes x {k})")
    return eigenmaps


def build_graph_cor(complete_time_series, similarity_delta=0.1):
    """
    Builds the correlation graph (graph_cor) for the NGSIM dataset.

    Args:
        complete_time_series (np.ndarray): Full time series (time_steps, num_nodes, num_features).
        similarity_delta (float): Threshold to keep correlations.

    Returns:
        np.ndarray: Asymmetric correlation graph (num_nodes, num_nodes)
    """

    print("Building real graph_cor based on Pearson correlation...")

    time_steps, num_nodes, num_features = complete_time_series.shape

    # Flatten (time_steps, features) into (time_steps, num_nodes*features) view
    feature_traces = complete_time_series.transpose(1, 0, 2).reshape(num_nodes, -1)  # (num_nodes, time_steps * features)

    # Compute Pearson correlation matrix
    cor_matrix = np.corrcoef(feature_traces)

    # Set weak correlations to 0 (optional thresholding)
    cor_matrix[np.abs(cor_matrix) < similarity_delta] = 0.0

    # No need to symmetrize: asymmetric allowed
    print(f"graph_cor built: {cor_matrix.shape} matrix with {np.count_nonzero(cor_matrix)} non-zero entries")

    return cor_matrix.astype(np.float32)



def add_self_loop(A):
    """
    Adds self-loops to adjacency matrix A.
    """
    B = A.copy()
    np.fill_diagonal(B, 1.0)
    return B

def row_normalize(A):
    """
    Row-normalizes a matrix A (sum of each row becomes 1).
    """
    A = A.astype(np.float32)
    rowsum = np.sum(A, axis=1)
    rowsum[rowsum == 0] = 1.0  # avoid division by zero
    return (A.T / rowsum).T


def build_transition_matrices(graph_conn, graph_sml, graph_cor=None):
    """
    Builds transition matrices from adjacency graphs.

    Args:
        graph_conn (np.ndarray): Physical adjacency matrix.
        graph_sml (np.ndarray): Feature similarity adjacency matrix.
        graph_cor (np.ndarray or None): Correlation graph. If None, dummy will be used.

    Returns:
        np.ndarray: Stacked transition matrices of shape (3, num_nodes, num_nodes)
    """

    print("Building transition matrices...")

    # Add self-loops before normalizing
    S_conn = row_normalize(add_self_loop(graph_conn))
    S_sml = row_normalize(add_self_loop(graph_sml))

    if graph_cor is not None:
        S_cor = row_normalize(add_self_loop(graph_cor))
    else:
        # Dummy graph_cor: use identity (only self transitions)
        num_nodes = graph_conn.shape[0]
        S_cor = np.eye(num_nodes, dtype=np.float32)

    # Stack the three transition matrices
    transition_matrices = np.stack((S_conn, S_sml, S_cor), axis=0)

    print(f"transition_matrices built: {transition_matrices.shape} (3 x num_nodes x num_nodes)")
    return transition_matrices


def build_restday_csv(df, output_dir):
    datetimes = cudf.to_datetime(df["Global_Time"]).rename("time")
    unique_date = pd.Timestamp(datetimes.min()).floor('D')
    restday_df = cudf.DataFrame({"time": [unique_date], "rest": [0]})
    restday_df = restday_df.to_pandas()
    restday_df.to_csv(os.path.join(output_dir, "restday.csv"), index=False)



# ---------- MAIN ----------
if __name__ == "__main__":
    title = "i-80"
    input_csv = f"Cleaned_NGSIM/cleaned_ngsim_{title}.csv"
    output_dir = f"graph_snapshots/{title}/"

    print("Loading data...")
    df = load_cleaned_location(input_csv)
    df["timestamp"] = df["Global_Time"]

    print("Defining segment IDs...")
    df = define_segments_cudf(df, segment_size=50.0)

    print("Aggregating Node Features...")
    agg_df = aggregate_node_features_cudf(df, time_col="Global_Time")

    print("Generating sequences (with stride = 10)...")
    data = generate_sequences_by_segments(agg_df, in_len=30, out_len=10, stride=10)
    
    print("Splitting into train / val / test...")
    stats = split_and_save_dataset(data, output_dir)

    print("Building complete_time_series for graph structures...")
    # Concatenate x and y
    z = np.concatenate((data['x'], data['y']), axis=1)  # (num_samples, in_len + out_len, num_nodes, num_features)
    complete_time_series = z.reshape(-1, data['x'].shape[2], data['x'].shape[3])

    # Optionally save complete_time_series
    with open(os.path.join(output_dir, "complete_time_series.pkl"), "wb") as f:
        pickle.dump(complete_time_series, f)

    print("Saved complete_time_series.pkl")
    print("\n\nBuilding graph structures...")

    # Build node_index
    node_ids = df["segment_id"].unique().to_pandas().sort_values().tolist()
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    # Build and save graph_conn
    graph_conn = build_graph_conn(df, node_index)
    with open(os.path.join(output_dir, f"graph_ngsim{title}_conn.pkl"), "wb") as f:
        pickle.dump(graph_conn, f)

    print(f"Saved graph_ngsim{title}_conn.pkl")

    # Build and save graph_sml
    graph_sml = build_graph_sml(complete_time_series, similarity_delta=0.1)
    with open(os.path.join(output_dir, f"graph_ngsim{title}_sml.pkl"), "wb") as f:
        pickle.dump(graph_sml, f)

    print(f"Saved graph_ngsim{title}_sml.pkl")

    # Build and save graph_cor
    graph_cor = build_graph_cor(complete_time_series, similarity_delta=0.1)
    with open(os.path.join(output_dir, f"graph_ngsim{title}_cor.pkl"), "wb") as f:
        pickle.dump(graph_cor, f)

    print(f"Saved graph_ngsim{title}_cor.pkl")

    # Build and save eigenmaps
    eigenmaps = build_eigenmaps(graph_conn, k=8)
    with open(os.path.join(output_dir, "eigenmaps.pkl"), "wb") as f:
        pickle.dump(eigenmaps, f)

    # Build and save transition matrices
    transition_matrices = build_transition_matrices(graph_conn, graph_sml, graph_cor)
    with open(os.path.join(output_dir, "transition_matrices.pkl"), "wb") as f:
        pickle.dump(transition_matrices, f)

    # Build and save restday.csv
    build_restday_csv(df, output_dir)

    print("\nFinal split counts:")
    for split, count in stats.items():
        print(f"  {split.capitalize()}: {count} samples")

    print("\nAll graph structures and auxiliary files built and saved!")

