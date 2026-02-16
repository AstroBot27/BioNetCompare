import io
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

st.set_page_config(
    page_title="BioNet Compare — Biological Network Comparison",
    page_icon="🧬",
    layout="wide",
)

# -----------------------------
# Download button key helper
# -----------------------------

def unique_dl_key(prefix: str) -> str:
    """Return a unique key for Streamlit widgets to avoid DuplicateElementId.

    Streamlit auto-generates widget IDs from (type + parameters). When we create
    multiple download buttons with similar parameters across tabs, Streamlit can
    raise StreamlitDuplicateElementId.

    We avoid this by giving every download_button a unique key.
    """
    if "_dl_counter" not in st.session_state:
        st.session_state["_dl_counter"] = 0
    st.session_state["_dl_counter"] += 1
    # Keep key short and safe
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", prefix)[:60]
    return f"dl_{safe}_{st.session_state['_dl_counter']}"


# -----------------------------
# Utilities
# -----------------------------

DEFAULT_GLOBAL_PROPS = [
    "n_nodes",
    "n_edges",
    "density",
    "avg_degree",
    "avg_clustering",
    "n_components",
]

DEFAULT_LOCAL_PROPS = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
]

@dataclass
class NetworkSpec:
    key: str
    display_name: str
    directed: bool
    has_weight: bool
    delimiter: str
    weight_col: Optional[int] = None
    source_col: int = 0
    target_col: int = 1


def _safe_name(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._\- ]+", "_", x).strip()
    return x if x else "network"


def read_network(file_bytes: bytes, filename: str, spec: NetworkSpec) -> nx.Graph:
    ext = "." + filename.lower().split(".")[-1]

    if ext in {".graphml", ".xml"}:
        bio = io.BytesIO(file_bytes)
        G = nx.read_graphml(bio)
        # Respect user's choice
        if spec.directed and not G.is_directed():
            G = nx.DiGraph(G)
        if (not spec.directed) and G.is_directed():
            G = nx.Graph(G)
        if spec.has_weight:
            for _, _, d in G.edges(data=True):
                if "weight" not in d:
                    d["weight"] = 1.0
        return G

    text = file_bytes.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    if ext == ".sif":
        G = nx.DiGraph() if spec.directed else nx.Graph()
        for ln in lines:
            parts = re.split(r"\s+", ln)
            if len(parts) < 3:
                continue
            src = parts[0]
            inter = parts[1]
            tgts = parts[2:]
            for t in tgts:
                if spec.has_weight:
                    G.add_edge(src, t, interaction=inter, weight=1.0)
                else:
                    G.add_edge(src, t, interaction=inter)
        return G

    # Edge list
    G = nx.DiGraph() if spec.directed else nx.Graph()
    delim = spec.delimiter

    for ln in lines:
        parts = ln.split(delim) if delim != "whitespace" else re.split(r"\s+", ln)
        parts = [p.strip() for p in parts if p.strip() != ""]
        if len(parts) < 2:
            continue
        u = parts[spec.source_col] if spec.source_col < len(parts) else parts[0]
        v = parts[spec.target_col] if spec.target_col < len(parts) else parts[1]

        attrs = {}
        if spec.has_weight:
            idx = spec.weight_col if spec.weight_col is not None else 2
            if idx < len(parts):
                try:
                    attrs["weight"] = float(parts[idx])
                except Exception:
                    attrs["weight"] = 1.0
            else:
                attrs["weight"] = 1.0

        G.add_edge(u, v, **attrs)

    return G


def sanity_check_graph(G: nx.Graph) -> Tuple[bool, List[str]]:
    msgs = []
    ok = True
    if G.number_of_nodes() == 0:
        ok = False
        msgs.append("❌ No nodes found.")
    if G.number_of_edges() == 0:
        ok = False
        msgs.append("❌ No edges found.")

    msgs.append(f"✅ Parsed graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")

    loops = list(nx.selfloop_edges(G))
    if loops:
        msgs.append(f"⚠️ Contains {len(loops):,} self-loop edges (kept as-is).")

    w_total = 0
    w_missing = 0
    for _, _, d in G.edges(data=True):
        if "weight" in d:
            w_total += 1
            if d.get("weight") is None:
                w_missing += 1
    if w_total and w_missing:
        msgs.append(f"⚠️ {w_missing:,}/{w_total:,} weighted edges have missing weight; treated as 1.0 during analysis.")

    return ok, msgs


def edge_set_for_jaccard(G: nx.Graph) -> Set[Tuple[str, str]]:
    if G.is_directed():
        return {(str(u), str(v)) for u, v in G.edges()}
    return {tuple(sorted((str(u), str(v)))) for u, v in G.edges()}


def compute_global_props(G: nx.Graph) -> Dict[str, float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    props = {
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": float(nx.density(G)) if n > 1 else np.nan,
    }

    # average degree
    if n > 0:
        degs = [d for _, d in (G.out_degree() if G.is_directed() else G.degree())]
        props["avg_degree"] = float(np.mean(degs)) if degs else np.nan
    else:
        props["avg_degree"] = np.nan

    # clustering
    try:
        H = G.to_undirected() if G.is_directed() else G
        props["avg_clustering"] = float(nx.average_clustering(H)) if n > 2 else np.nan
    except Exception:
        props["avg_clustering"] = np.nan

    # components
    try:
        H = G.to_undirected() if G.is_directed() else G
        props["n_components"] = float(nx.number_connected_components(H)) if n else np.nan
        if n:
            comps = list(nx.connected_components(H))
            props["largest_component_frac"] = float(max(len(c) for c in comps) / n) if comps else np.nan
        else:
            props["largest_component_frac"] = np.nan
    except Exception:
        props["n_components"] = np.nan
        props["largest_component_frac"] = np.nan

    if G.is_directed():
        try:
            props["n_strong_components"] = float(nx.number_strongly_connected_components(G))
        except Exception:
            props["n_strong_components"] = np.nan

    # path-based on LCC
    try:
        H = G.to_undirected() if G.is_directed() else G
        if H.number_of_nodes() > 1 and H.number_of_edges() > 0:
            largest = max(nx.connected_components(H), key=len)
            S = H.subgraph(largest).copy()
            if S.number_of_nodes() > 1:
                props["avg_shortest_path_lcc"] = float(nx.average_shortest_path_length(S))
                try:
                    props["diameter_lcc"] = float(nx.diameter(S))
                except Exception:
                    props["diameter_lcc"] = np.nan
            else:
                props["avg_shortest_path_lcc"] = np.nan
                props["diameter_lcc"] = np.nan
        else:
            props["avg_shortest_path_lcc"] = np.nan
            props["diameter_lcc"] = np.nan
    except Exception:
        props["avg_shortest_path_lcc"] = np.nan
        props["diameter_lcc"] = np.nan

    return props


def compute_local_props(G: nx.Graph) -> pd.DataFrame:
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame()

    df = pd.DataFrame(index=pd.Index([str(n) for n in nodes], name="node"))

    if G.is_directed():
        df["in_degree"] = pd.Series({str(n): float(d) for n, d in G.in_degree()})
        df["out_degree"] = pd.Series({str(n): float(d) for n, d in G.out_degree()})
        df["degree"] = df["in_degree"] + df["out_degree"]
    else:
        df["degree"] = pd.Series({str(n): float(d) for n, d in G.degree()})

    H = G.to_undirected() if G.is_directed() else G

    try:
        df["betweenness"] = pd.Series(nx.betweenness_centrality(H, normalized=True))
    except Exception:
        df["betweenness"] = np.nan

    try:
        df["closeness"] = pd.Series(nx.closeness_centrality(H))
    except Exception:
        df["closeness"] = np.nan

    try:
        df["eigenvector"] = pd.Series(nx.eigenvector_centrality_numpy(H))
    except Exception:
        try:
            df["eigenvector"] = pd.Series(nx.eigenvector_centrality(H, max_iter=2000))
        except Exception:
            df["eigenvector"] = np.nan

    try:
        df["clustering"] = pd.Series(nx.clustering(H))
    except Exception:
        df["clustering"] = np.nan

    if G.is_directed():
        try:
            df["pagerank"] = pd.Series(nx.pagerank(G, weight="weight"))
        except Exception:
            df["pagerank"] = np.nan

    return df.sort_index()


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pairwise_jaccard(sets: Dict[str, Set]) -> pd.DataFrame:
    keys = list(sets.keys())
    mat = np.zeros((len(keys), len(keys)), dtype=float)
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            mat[i, j] = jaccard(sets[k1], sets[k2])
    return pd.DataFrame(mat, index=keys, columns=keys)


def download_df(df: pd.DataFrame, filename_base: str, sep: str = ","):
    data = df.to_csv(index=True, sep=sep).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download ({'CSV' if sep == ',' else 'TSV'})",
        data=data,
        file_name=f"{filename_base}.{'csv' if sep == ',' else 'tsv'}",
        mime="text/csv" if sep == "," else "text/tab-separated-values",
        key=unique_dl_key(f"df_{filename_base}_{'csv' if sep == ',' else 'tsv'}"),
    )


def download_plotly_fig(fig: go.Figure, filename_base: str):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.download_button(
            "⬇️ HTML",
            data=fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8"),
            file_name=f"{filename_base}.html",
            mime="text/html",
            key=unique_dl_key(f"fig_{filename_base}_html"),
        )
    with c2:
        try:
            st.download_button(
                "⬇️ PNG",
                data=fig.to_image(format="png", scale=2),
                file_name=f"{filename_base}.png",
                mime="image/png",
                key=unique_dl_key(f"fig_{filename_base}_png"),
            )
        except Exception:
            st.caption("PNG export unavailable (install 'kaleido').")
    with c3:
        try:
            st.download_button(
                "⬇️ JPG",
                data=fig.to_image(format="jpg", scale=2),
                file_name=f"{filename_base}.jpg",
                mime="image/jpeg",
                key=unique_dl_key(f"fig_{filename_base}_jpg"),
            )
        except Exception:
            st.caption("JPG export unavailable (install 'kaleido').")
    with c4:
        try:
            st.download_button(
                "⬇️ PDF",
                data=fig.to_image(format="pdf"),
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                key=unique_dl_key(f"fig_{filename_base}_pdf"),
            )
        except Exception:
            st.caption("PDF export unavailable (install 'kaleido').")
    with c5:
        try:
            st.download_button(
                "⬇️ SVG",
                data=fig.to_image(format="svg"),
                file_name=f"{filename_base}.svg",
                mime="image/svg+xml",
                key=unique_dl_key(f"fig_{filename_base}_svg"),
            )
        except Exception:
            st.caption("SVG export unavailable (install 'kaleido').")


# -----------------------------
# Session state
# -----------------------------

if "uploads" not in st.session_state:
    st.session_state.uploads = {}
if "selected_keys" not in st.session_state:
    st.session_state.selected_keys = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis" not in st.session_state:
    st.session_state.analysis = {}


# -----------------------------
# Header
# -----------------------------

st.title("🧬 BioNet Compare — Compare Multiple Biological Networks")
st.caption("Upload multiple networks (Edge list / SIF / GraphML), sanity-check them, then run analysis and explore results in tabs.")


# -----------------------------
# Upload + Configure + Select
# -----------------------------

st.subheader("1) Upload networks")

with st.expander("Supported formats & tips", expanded=False):
    st.markdown(
        """
**Formats**
- **Edge list** (`.tsv`, `.csv`, `.txt`): at least two columns: `source` and `target`. Optional third column for **weight**.
- **SIF** (`.sif`): `source  interaction  target1  target2 ...` (weights are treated as 1.0).
- **GraphML** (`.graphml`/`.xml`): weights read from the `weight` edge attribute if present.

**Best practice**
- Use consistent node identifiers (e.g., gene symbols) across files.
- For undirected networks, order of edge endpoints does not matter.
        """
    )

uploads = st.file_uploader(
    "Upload one or more network files",
    type=["txt", "tsv", "csv", "sif", "graphml", "xml"],
    accept_multiple_files=True,
)

# ingest uploads
if uploads:
    for up in uploads:
        key = f"{up.name}:{up.size}"
        if key not in st.session_state.uploads:
            ext = "." + up.name.lower().split(".")[-1]
            default_delim = "\t" if ext == ".tsv" else ("," if ext == ".csv" else "whitespace")
            spec = NetworkSpec(
                key=key,
                display_name=_safe_name(up.name.rsplit(".", 1)[0]),
                directed=False,
                has_weight=False,
                delimiter=default_delim,
            )
            st.session_state.uploads[key] = {"file": up, "spec": spec, "graph": None, "ok": False, "msgs": []}
            if key not in st.session_state.selected_keys:
                st.session_state.selected_keys.append(key)
            st.session_state.analysis_done = False

# remove missing
current_keys = {f"{up.name}:{up.size}" for up in uploads} if uploads else set()
for k in list(st.session_state.uploads.keys()):
    if uploads is not None and k not in current_keys:
        st.session_state.uploads.pop(k, None)
        if k in st.session_state.selected_keys:
            st.session_state.selected_keys.remove(k)
        st.session_state.analysis_done = False

if not st.session_state.uploads:
    st.info("Upload network files to begin.")
    st.stop()

st.subheader("2) Configure & sanity-check")

for k, item in st.session_state.uploads.items():
    spec: NetworkSpec = item["spec"]
    up = item["file"]
    ext = "." + up.name.lower().split(".")[-1]

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([2.2, 1.2, 1.2, 1.4, 1.6])
        with c1:
            spec.display_name = st.text_input("Network name", value=spec.display_name, key=f"name_{k}")
        with c2:
            spec.directed = st.toggle("Directed", value=spec.directed, key=f"dir_{k}")
        with c3:
            spec.has_weight = st.toggle("Has weight", value=spec.has_weight, key=f"w_{k}")
        with c4:
            if ext in {".csv", ".tsv", ".txt"}:
                spec.delimiter = st.selectbox("Delimiter", options=["\t", ",", "whitespace"],
                                              index=["\t", ",", "whitespace"].index(spec.delimiter) if spec.delimiter in ["\t", ",", "whitespace"] else 2,
                                              key=f"delim_{k}")
            else:
                st.caption(f"Format: {ext.upper()} (delimiter not applicable)")
        with c5:
            if ext in {".csv", ".tsv", ".txt"} and spec.has_weight:
                spec.weight_col = st.number_input("Weight col (0-based)", min_value=2, value=int(spec.weight_col) if spec.weight_col is not None else 2, step=1, key=f"wcol_{k}")
            else:
                spec.weight_col = None
                st.caption(" ")

        b1, b2 = st.columns([1, 6])
        with b1:
            parse_clicked = st.button("Sanity check", key=f"check_{k}")
        with b2:
            st.caption(f"File: **{up.name}** ({up.size/1024:.1f} KB)")

        if parse_clicked or (item["graph"] is None):
            try:
                G = read_network(up.getvalue(), up.name, spec)
                if spec.has_weight:
                    for _, _, d in G.edges(data=True):
                        if d.get("weight") is None:
                            d["weight"] = 1.0
                        try:
                            d["weight"] = float(d.get("weight", 1.0))
                        except Exception:
                            d["weight"] = 1.0

                ok, msgs = sanity_check_graph(G)
                item["graph"] = G
                item["ok"] = ok
                item["msgs"] = msgs
                st.session_state.analysis_done = False
            except Exception as e:
                item["graph"] = None
                item["ok"] = False
                item["msgs"] = [f"❌ Failed to parse: {e}"]
                st.session_state.analysis_done = False

        for m in item["msgs"]:
            st.write(m)

st.subheader("3) Select networks for comparison")
sel_cols = st.columns(min(4, len(st.session_state.uploads)))
all_keys = list(st.session_state.uploads.keys())
for i, k in enumerate(all_keys):
    item = st.session_state.uploads[k]
    spec: NetworkSpec = item["spec"]
    with sel_cols[i % len(sel_cols)]:
        checked = st.checkbox(spec.display_name, value=(k in st.session_state.selected_keys), key=f"sel_{k}", disabled=not item["ok"])
        if checked and k not in st.session_state.selected_keys:
            st.session_state.selected_keys.append(k)
            st.session_state.analysis_done = False
        if (not checked) and k in st.session_state.selected_keys:
            st.session_state.selected_keys.remove(k)
            st.session_state.analysis_done = False

selected = st.session_state.selected_keys
if len(selected) < 2:
    st.warning("Select at least **two** valid networks for comparison.")

all_selected_ok = (len(selected) >= 2) and all(st.session_state.uploads[k]["ok"] for k in selected)

st.subheader("4) Run analysis")
run = st.button("🚀 Perform analysis", disabled=not all_selected_ok)


# -----------------------------
# Analysis pipeline (precompute on landing page)
# -----------------------------

def run_analysis(selected_keys: List[str]):
    uploads = st.session_state.uploads

    graphs: Dict[str, nx.Graph] = {}
    names: Dict[str, str] = {}

    for k in selected_keys:
        spec: NetworkSpec = uploads[k]["spec"]
        G = uploads[k]["graph"]
        if G is None:
            continue
        graphs[k] = G
        names[k] = spec.display_name

    log_lines = []

    p1 = st.progress(0.0, text="Global property comparison: waiting")
    p2 = st.progress(0.0, text="Local property comparison: waiting")
    p3 = st.progress(0.0, text="Node & edge venn/jaccard: waiting")
    p4 = st.progress(0.0, text="Union network comparison: waiting")

    # 1) Global
    p1.progress(0.05, text="Global property comparison: computing global metrics")
    global_rows = []
    for i, (k, G) in enumerate(graphs.items()):
        props = compute_global_props(G)
        props["network"] = names[k]
        props["directed"] = bool(G.is_directed())
        global_rows.append(props)
        p1.progress(0.05 + 0.9 * (i + 1) / max(1, len(graphs)), text=f"Global property comparison: {names[k]}")
        log_lines.append(f"[Global] {names[k]}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    global_df = pd.DataFrame(global_rows).set_index("network").sort_index()
    p1.progress(1.0, text="Global property comparison: done")

    # 2) Local
    p2.progress(0.05, text="Local property comparison: computing node-level metrics")
    local_dfs = {}
    for i, (k, G) in enumerate(graphs.items()):
        df = compute_local_props(G)
        local_dfs[k] = df
        p2.progress(0.05 + 0.9 * (i + 1) / max(1, len(graphs)), text=f"Local property comparison: {names[k]}")
        log_lines.append(f"[Local] {names[k]}: computed {df.shape[1]} properties for {df.shape[0]} nodes")
    p2.progress(1.0, text="Local property comparison: done")

    # 3) Jaccard matrices
    p3.progress(0.10, text="Node & edge venn/jaccard: building node/edge sets")
    node_sets = {names[k]: {str(n) for n in graphs[k].nodes()} for k in graphs}
    edge_sets = {names[k]: edge_set_for_jaccard(graphs[k]) for k in graphs}
    node_j = pairwise_jaccard(node_sets)
    edge_j = pairwise_jaccard(edge_sets)
    p3.progress(1.0, text="Node & edge venn/jaccard: done")
    log_lines.append("[Venn/Jaccard] computed pairwise node and edge Jaccard matrices")

    # 4) Union
    p4.progress(0.10, text="Union network comparison: building union network")
    union_directed = any(graphs[k].is_directed() for k in graphs)
    U = nx.DiGraph() if union_directed else nx.Graph()
    for _, G in graphs.items():
        U.add_nodes_from([str(n) for n in G.nodes()])
        for u, v in G.edges():
            U.add_edge(str(u), str(v))

    p4.progress(0.55, text="Union network comparison: computing distance from union")
    union_nodes = {str(n) for n in U.nodes()}
    union_edges = edge_set_for_jaccard(U)

    dist_rows = []
    for i, (k, G) in enumerate(graphs.items()):
        nset = {str(n) for n in G.nodes()}
        eset = edge_set_for_jaccard(G)
        node_jacc = jaccard(nset, union_nodes)
        edge_jacc = jaccard(eset, union_edges)
        dist_rows.append({
            "network": names[k],
            "node_jaccard_to_union": node_jacc,
            "edge_jaccard_to_union": edge_jacc,
            "node_distance": 1 - node_jacc,
            "edge_distance": 1 - edge_jacc,
        })
        p4.progress(0.55 + 0.4 * (i + 1) / max(1, len(graphs)), text=f"Union network comparison: {names[k]}")

    union_dist_df = pd.DataFrame(dist_rows).set_index("network").sort_index()
    p4.progress(1.0, text="Union network comparison: done")
    log_lines.append(f"[Union] union has {U.number_of_nodes()} nodes and {U.number_of_edges()} edges")

    return {
        "graphs": graphs,
        "names": names,
        "global_df": global_df,
        "local_dfs": local_dfs,
        "node_jaccard": node_j,
        "edge_jaccard": edge_j,
        "union_graph": U,
        "union_dist_df": union_dist_df,
        "node_sets": node_sets,
        "edge_sets": edge_sets,
        "log": log_lines,
    }


if run:
    st.session_state.analysis_done = False
    st.session_state.analysis = {}
    st.markdown("---")
    st.subheader("Running analysis")
    with st.spinner("Computing metrics... this may take a while for large networks."):
        result = run_analysis(selected)
    st.session_state.analysis = result
    st.session_state.analysis_done = True

    st.success("✅ Analysis is done — you can now view the results in the tabs below.")
    with st.expander("Analysis log", expanded=False):
        for ln in result["log"]:
            st.code(ln)

if not st.session_state.analysis_done:
    st.info("After sanity-checking and selecting networks, click **Perform analysis**.")
    st.stop()

A = st.session_state.analysis

# -----------------------------
# Results tabs
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "1) Global properties",
    "2) Local properties",
    "3) Node & edge Venn/Jaccard",
    "4) Union network comparison",
])

# ---- Tab 1: Global ----
with tab1:
    st.header("Global property comparison")
    global_df: pd.DataFrame = A["global_df"].copy()

    all_props = [c for c in global_df.columns if c not in {"directed"}]
    default = [p for p in DEFAULT_GLOBAL_PROPS if p in all_props]
    selected_props = st.multiselect("Select global properties to compare", options=all_props, default=default)

    show_df = global_df[selected_props + (["directed"] if "directed" in global_df.columns else [])].copy()
    st.subheader("Table")
    st.dataframe(show_df, use_container_width=True)
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        download_df(show_df, "global_properties", sep=",")
    with cdl2:
        download_df(show_df, "global_properties", sep="\t")

    st.subheader("Plots")
    if selected_props:
        long = show_df[selected_props].reset_index().melt(id_vars="network", var_name="property", value_name="value")
        fig = px.bar(long, x="network", y="value", color="property", barmode="group", title="Global properties across networks")
        fig.update_layout(xaxis_title="Network", yaxis_title="Value", legend_title="Property")
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_fig(fig, "global_properties_bar")

        st.markdown("**Radar (normalized per property)**")
        norm = show_df[selected_props].copy()
        for c in norm.columns:
            col = norm[c].astype(float)
            mn, mx = np.nanmin(col.values), np.nanmax(col.values)
            norm[c] = (col - mn) / (mx - mn) if np.isfinite(mn) and np.isfinite(mx) and mx > mn else 0.0

        radar = go.Figure()
        cats = list(norm.columns)
        for net, row in norm.iterrows():
            radar.add_trace(go.Scatterpolar(r=np.r_[row.values, row.values[0]], theta=cats + [cats[0]], name=net, fill="toself", opacity=0.45))
        radar.update_layout(title="Global properties (min-max normalized)", polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(radar, use_container_width=True)
        download_plotly_fig(radar, "global_properties_radar")

# ---- Tab 2: Local ----
with tab2:
    st.header("Local property comparison")

    local_dfs_by_name = {A["names"][k]: A["local_dfs"][k] for k in A["local_dfs"]}

    prop_union = sorted(set().union(*[set(df.columns) for df in local_dfs_by_name.values()])) if local_dfs_by_name else []
    default = [p for p in DEFAULT_LOCAL_PROPS if p in prop_union]

    selected_props = st.multiselect("Select local properties to compare", options=prop_union, default=default)
    k = st.number_input("Top-k nodes per property (per network)", min_value=1, max_value=500, value=10, step=1)
    show_all = st.checkbox("Show all nodes (can be large)", value=False)
    query = st.text_input("Search node (regex supported)", value="")

    st.subheader("Union node table")
    union_nodes = sorted(set().union(*[set(df.index) for df in local_dfs_by_name.values()])) if local_dfs_by_name else []
    if (not show_all) and len(union_nodes) > 5000:
        st.warning(f"There are {len(union_nodes):,} union nodes. For performance, enable 'Show all nodes' or use search.")
        union_nodes = union_nodes[:5000]

    big = pd.DataFrame(index=pd.Index(union_nodes, name="node"))
    topk_flags = {}

    for net, df in local_dfs_by_name.items():
        if df.empty:
            continue
        for prop in selected_props:
            colname = f"{net}::{prop}"
            big[colname] = df[prop].reindex(big.index) if prop in df.columns else np.nan
        for prop in selected_props:
            if prop in df.columns:
                top_nodes = df[prop].dropna().sort_values(ascending=False).head(int(k)).index.tolist()
                topk_flags[(net, prop)] = set(top_nodes)

    if query.strip():
        try:
            pat = re.compile(query)
            view = big.loc[[bool(pat.search(n)) for n in big.index]].copy()
        except re.error:
            st.error("Invalid regex. Showing unfiltered table.")
            view = big.copy()
    else:
        view = big.copy()

    def _style_topk(s: pd.Series):
        if "::" not in s.name:
            return ["" for _ in s]
        net, prop = s.name.split("::", 1)
        tops = topk_flags.get((net, prop), set())
        return ["font-weight: 700; background-color: #fff3cd" if idx in tops else "" for idx in s.index]

    st.dataframe(view.style.apply(_style_topk, axis=0), use_container_width=True, height=420)
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        download_df(view, "local_properties_union_table", sep=",")
    with cdl2:
        download_df(view, "local_properties_union_table", sep="\t")

    st.subheader("Top-k nodes per property (summary)")
    rows = [{"network": net, "property": prop, "top_k_nodes": ", ".join(list(tops)[:50]) + (" ..." if len(tops) > 50 else "")}
            for (net, prop), tops in topk_flags.items()]
    topk_df = pd.DataFrame(rows).sort_values(["property", "network"]) if rows else pd.DataFrame(columns=["network", "property", "top_k_nodes"])
    st.dataframe(topk_df, use_container_width=True)
    download_df(topk_df, "local_properties_topk_summary", sep=",")

    st.subheader("Heatmap (current table view)")
    if view.shape[0] == 0 or view.shape[1] == 0:
        st.info("Nothing to plot.")
    else:
        heat = view.copy()
        for c in heat.columns:
            col = heat[c].astype(float)
            mn, mx = np.nanmin(col.values), np.nanmax(col.values)
            heat[c] = (col - mn) / (mx - mn) if np.isfinite(mn) and np.isfinite(mx) and mx > mn else 0.0

        max_rows = 500
        if heat.shape[0] > max_rows and not show_all:
            st.caption(f"Heatmap shows first {max_rows} rows for performance. Use search to focus.")
            heat = heat.iloc[:max_rows]

        fig = px.imshow(heat, aspect="auto", color_continuous_scale="Viridis", title="Local property heatmap (min-max normalized per column)")
        fig.update_layout(xaxis_title="Network::Property", yaxis_title="Node")
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_fig(fig, "local_properties_heatmap")

# ---- Tab 3: Venn/Jaccard ----
with tab3:
    st.header("Node & edge overlap (Jaccard + Venn)")

    node_j: pd.DataFrame = A["node_jaccard"].copy()
    edge_j: pd.DataFrame = A["edge_jaccard"].copy()

    st.subheader("Pairwise Jaccard indices")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Node Jaccard**")
        st.dataframe(node_j, use_container_width=True)
        download_df(node_j, "node_jaccard_matrix", sep=",")
        fig = px.imshow(node_j, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Node Jaccard heatmap")
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_fig(fig, "node_jaccard_heatmap")
    with c2:
        st.markdown("**Edge Jaccard**")
        st.dataframe(edge_j, use_container_width=True)
        download_df(edge_j, "edge_jaccard_matrix", sep=",")
        fig = px.imshow(edge_j, text_auto=True, aspect="auto", color_continuous_scale="Purples", title="Edge Jaccard heatmap")
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_fig(fig, "edge_jaccard_heatmap")

    st.subheader("Venn (up to 3 networks)")
    nets = list(A["node_sets"].keys())
    subset = st.multiselect("Choose 2–3 networks", options=nets, default=nets[:3], help="Venn diagrams are shown for 2 or 3 networks.")

    if len(subset) < 2:
        st.info("Select at least 2 networks.")
    elif len(subset) > 3:
        st.warning("Please select at most 3 networks for Venn diagrams.")
    else:
        node_sets = {k: A["node_sets"][k] for k in subset}
        edge_sets = {k: A["edge_sets"][k] for k in subset}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Node Venn**")
            fig, ax = plt.subplots(figsize=(5, 4))
            if len(subset) == 2:
                a, b = subset
                venn2([node_sets[a], node_sets[b]], set_labels=(a, b), ax=ax)
            else:
                a, b, c = subset
                venn3([node_sets[a], node_sets[b], node_sets[c]], set_labels=(a, b, c), ax=ax)
            ax.set_title("Node overlap")
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
            st.download_button("⬇️ Download PNG", data=buf.getvalue(), file_name="node_venn.png", mime="image/png",
                               key=unique_dl_key("venn_nodes_png"))

        with c2:
            st.markdown("**Edge Venn**")
            fig, ax = plt.subplots(figsize=(5, 4))
            if len(subset) == 2:
                a, b = subset
                venn2([edge_sets[a], edge_sets[b]], set_labels=(a, b), ax=ax)
            else:
                a, b, c = subset
                venn3([edge_sets[a], edge_sets[b], edge_sets[c]], set_labels=(a, b, c), ax=ax)
            ax.set_title("Edge overlap")
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
            st.download_button("⬇️ Download PNG", data=buf.getvalue(), file_name="edge_venn.png", mime="image/png",
                               key=unique_dl_key("venn_edges_png"))

# ---- Tab 4: Union ----
with tab4:
    st.header("Comparison with respect to the union network")

    union_dist: pd.DataFrame = A["union_dist_df"].copy()

    st.subheader("Distance to union (1 - Jaccard)")
    st.dataframe(union_dist, use_container_width=True)
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        download_df(union_dist, "distance_to_union", sep=",")
    with cdl2:
        download_df(union_dist, "distance_to_union", sep="\t")

    st.subheader("Radar plots")
    nets = union_dist.index.tolist()
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        r = union_dist["node_distance"].values
        fig.add_trace(go.Scatterpolar(r=np.r_[r, r[0]], theta=nets + [nets[0]], fill="toself", name="Node distance"))
        fig.update_layout(title="Node distance from union", polar=dict(radialaxis=dict(visible=True, range=[0, max(0.01, float(np.nanmax(r)))])), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_fig(fig, "union_node_distance_radar")

    with c2:
        fig = go.Figure()
        r = union_dist["edge_distance"].values
        fig.add_trace(go.Scatterpolar(r=np.r_[r, r[0]], theta=nets + [nets[0]], fill="toself", name="Edge distance"))
        fig.update_layout(title="Edge distance from union", polar=dict(radialaxis=dict(visible=True, range=[0, max(0.01, float(np.nanmax(r)))])), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_fig(fig, "union_edge_distance_radar")

    st.subheader("Union network summary")
    U: nx.Graph = A["union_graph"]
    st.write(f"Union network has **{U.number_of_nodes():,} nodes** and **{U.number_of_edges():,} edges**. Directed: **{U.is_directed()}**")
