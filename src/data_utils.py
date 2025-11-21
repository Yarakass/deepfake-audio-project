from pathlib import Path
import pandas as pd

def load_asvspoof_protocol(proto_path: str):
    """Lit le protocole ASVspoof (format LA_0079 LA_T_xxx - - bonafide/spoof) et renvoie un DataFrame (utt, is_spoof)."""
    df = pd.read_csv(
        proto_path,
        sep=r"\s+",
        header=None,
        names=["spk", "utt", "dash1", "dash2", "label"]  # <-- 5 colonnes
    )
    df["is_spoof"] = (df["label"].str.lower() == "spoof").astype(int)
    return df[["utt", "is_spoof"]]
def attach_paths_by_utt(df: pd.DataFrame, audio_root: str):
    """Associe à chaque utt le chemin du fichier audio correspondant."""
    root = Path(audio_root)
    files = {}
    for p in root.iterdir():
        if p.is_file():
            files[p.stem] = str(p)
    out = df.copy()
    out["path"] = out["utt"].map(files)
    out = out.dropna(subset=["path"])
    return out

def index_wavefake(root: str):
    """Construit un DataFrame path/is_spoof à partir de dossiers data/wavefake/real et data/wavefake/fake."""
    rootp = Path(root)
    rows = []
    for lbl, target in [("real",0),("fake",1)]:
        for p in (rootp/lbl).rglob("*.wav"):
            rows.append({"path":str(p), "is_spoof":target})
    return pd.DataFrame(rows)
