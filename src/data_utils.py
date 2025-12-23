from pathlib import Path
import pandas as pd

def load_asvspoof_protocol(proto_path: str):
    """Reads the ASVspoof protocol (format LA_0079 LA_T_xxx - - bonafide/spoof) and returns a DataFrame (utt, is_spoof)"""
    df = pd.read_csv(
        proto_path,
        sep=r"\s+",
        header=None,
        names=["spk", "utt", "dash1", "dash2", "label"]
    )
    df["is_spoof"] = (df["label"].str.lower() == "spoof").astype(int)
    return df[["utt", "is_spoof"]]
def attach_paths_by_utt(df: pd.DataFrame, audio_root: str):
    """Associate each utt with the path to the corresponding audio file."""
    root = Path(audio_root)
    files = {}
    for p in root.iterdir():
        if p.is_file():
            files[p.stem] = str(p)
    out = df.copy()
    out["path"] = out["utt"].map(files)
    out = out.dropna(subset=["path"])
    return out
