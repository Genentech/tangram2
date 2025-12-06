import pandas as pd

def read_signature(sig_pth : str, filter_uniprot: bool = False):
    sig = pd.read_csv(sig_pth, header =None, sep = '\t')
    sig.columns = ['annot','gene']
    
    if filter_uniprot:
        sig = sig.iloc[sig.annot.str.startswith('UniProt').values]
    sig = sig.iloc[:,1].values.tolist()
    
    return sig