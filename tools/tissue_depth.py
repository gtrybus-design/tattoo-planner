"""Facial soft tissue depths (skin + fat + muscle over the skull) at the standard craniofacial landmarks.

The values are computed here, at build time, from the C-Table: the per-person data compiled alongside the
T-Tables (assets/tissue-depth/C-Table_v2018_0.csv, from https://www.craniofacialidentification.com/Data.html;
19 studies, 1971 people). As in the T-Tables, the depth at each landmark is the mean over adult men (18 and over;
left and right sides pooled), and for the skewed landmarks (gonion, mid-ramus, menton, supramentale = the
mentolabial sulcus) the shorth (the mean of the shortest half of the data) instead.

These are our own pooled values from the C-Table, not copied from the published T-Tables (whose papers are
behind paywalls). The T-Tables' terms of use ask for these references:
  Stephan CN, Simpson EK (2008). Facial soft tissue depths in craniofacial identification (part I): an analytical
    review of the published adult data. Journal of Forensic Sciences 53(6):1257-1272. doi:10.1111/j.1556-4029.2008.00852.x
  Stephan CN (2014). The application of the central limit theorem and the law of large numbers to facial soft tissue
    depths: T-Table robustness and trends since 2008. Journal of Forensic Sciences 59(2):454-462. doi:10.1111/1556-4029.12328
  Stephan CN (2017). 2018 tallied facial soft tissue thicknesses for adults and sub-adults. Forensic Science
    International 280:113-123. doi:10.1016/j.forsciint.2017.09.016
  Hona TWPT, Stephan CN (2024). Global facial soft tissue thicknesses for craniofacial identification (2023): a review
    of 140 years of data since Welcker's first study. International Journal of Legal Medicine. doi:10.1007/s00414-023-03087-x
  and the original studies listed with the T-Tables (https://www.craniofacialidentification.com/Refs.html).
"""
import csv
import numpy as np

# landmark code -> name (as in the C-Table); bilateral ones are measured on both sides
MIDLINE = {'op': 'opisthocranion', 'v': 'vertex', 'm': 'metopion', 'sg': 'supraglabella', 'g': 'glabella',
           'n': 'nasion', 'mn': 'mid-nasal', 'rhi': 'rhinion', 'sn': 'subnasale', 'mp': 'mid-philtrum',
           'ls': 'labrale superius', 'li': 'labrale inferius', 'sm': 'supramentale', 'pg': 'pogonion',
           'gn': 'gnathion', 'me': 'menton'}
BILATERAL = {'mso': 'mid-supraorbital', 'mio': 'mid-infraorbital', 'ac': 'alar curvature', 'go': 'gonion',
             'zy': 'zygion', 'sC': 'supra canine', 'iC': 'infra canine', 'sM2': 'supra 2nd molar',
             'iM2': 'infra 2nd molar', 'mr': 'mid-ramus', 'mmb': 'mid-mandibular border'}
SKEWED = {'go', 'mr', 'me', 'sm'}      # (shorth instead of the mean)


def shorth(v):
    """The mean of the shortest half of the data (Andrews et al. 1972)."""
    v = np.sort(v); h = len(v) // 2 + 1
    k = np.argmin(v[h - 1:] - v[:len(v) - h + 1])
    return v[k:k + h].mean()


def depths(path, sex='M', min_age=18):
    """{code: (depth in metres, number of people, 'mean' or 'shorth')} for every landmark with data."""
    rows = list(csv.reader(open(path, encoding='latin-1')))
    hdr = rows[2]; col = {h: i for i, h in enumerate(hdr) if h}
    out = {}
    for code in list(MIDLINE) + list(BILATERAL):
        cols = [col[code]] if code in MIDLINE else [col['L_' + code], col['R_' + code]]
        vals = []
        for r in rows[3:]:
            try: age = float(r[col['Age (yrs)']])
            except ValueError: continue
            if r[col['Sex']] != sex or age < min_age: continue
            for c in cols:
                if r[c].strip():
                    try: vals.append(float(r[c]))
                    except ValueError: pass
        if not vals: continue
        v = np.array(vals)
        stat = 'shorth' if code in SKEWED else 'mean'
        out[code] = ((shorth(v) if code in SKEWED else v.mean()) / 1000, len(v), stat)
    return out


if __name__ == '__main__':
    import sys
    for k, (d, n, st) in depths(sys.argv[1]).items():
        print(f'{k:4s} {MIDLINE.get(k, BILATERAL.get(k)):24s} {d * 1000:5.1f} mm  ({st}, n={n})')
