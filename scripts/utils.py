def clean_song_name(filename):
    nfn = filename.replace('___', ' & ')
    nfn = nfn.replace('_t_', "'t ")
    nfn = nfn.replace('_t.', "'t.")
    nfn = nfn.replace('_s_', "'s ")
    nfn = nfn.replace('_s.', "'s.")
    nfn = nfn.replace('_ll', "'ll")
    nfn = nfn.replace('_ve_', "'ve ")
    nfn = nfn.replace('_re_', "'re ")
    nfn = nfn.replace('__', ' ')
    nfn = nfn.replace('_', ' ')
    if nfn.endswith('.mp3'):
        nfn = nfn[:-4]
    return nfn
