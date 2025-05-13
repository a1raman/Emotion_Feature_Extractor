# au_config.py (25.04.22 kh)

AU_DESCRIPTION_DICT = {
    "AU01": ["Inner brow raiser", "Frown", "Eyebrow raised", "Head lifting wrinkles", "Lift eyebrows"],
    "AU02": ["Outer brow raiser", "Outer brow lift", "Elevate outer brow", "Outer brow arch"],
    "AU04": ["Brow Lowerer", "Frowns furrowed", "Lower eyebrows", "A look of disapproval"],
    "AU05": ["Upper Lid Raiser", "Pupil enlargement", "Eyes widened", "Lift upper eyelids", "Raise upper eyelids"],
    "AU06": ["Cheek Raiser", "Smile", "Pleasure", "Eyes narrowing", "Slightly lower eyebrows"],
    "AU07": ["Lid Tightener", "Facial tightness", "Tightening of eyelids"],
    "AU09": ["Nose Wrinkler", "Wrinkle the nose", "Curl the nose", "Make a face", "Pucker the nose"],
    "AU10": ["Upper Lip Raiser", "Curl the lips upwards", "Upper lip lift", "Lips apart showing teeth"],
    "AU12": ["Lip Corner Puller", "Toothy smile", "Grinning", "Big smile", "Show teeth"],
    "AU14": ["Dimpler", "Cheek dimple", "Indentation when smiling", "Hollow on the face when smiling"],
    "AU15": ["Lip Corner Depressor", "Downturned corners of the mouth", "Downward mouth curvature", "Lower Lip Depressor"],
    "AU17": ["Chin Raiser", "Lift the chin", "Chin held high", "Lips arching", "Lips forming an upward curve"],
    "AU20": ["Lip stretcher", "Tense lips stretched", "Anxiously stretched lips", "Nasal flaring", "Nostrils enlarge"],
    "AU23": ["Lip Tightener", "Tighten the lips", "Purse the lips", "Press the lips together"],
    "AU25": ["Lips part", "Open the lips", "Slightly puzzled", "Lips slightly parted"],
    "AU26": ["Jaw Drop", "Mouth Stretch", "Open mouth wide", "Wide-mouthed", "Lips elongated"],
    "AU28": ["Lip Suck", "Purse lips", "Pucker lips", "Draw in lips", "Bring lips together"]
}


EMOTION_AU_RULES = {
    "happy": {"AU06", "AU12", "AU14"},
    "angry": {"AU04", "AU05", "AU07", "AU23", "AU10", "AU17"},
    "worried": {"AU28", "AU20"},
    "surprise": {"AU01", "AU02", "AU05", "AU26"},
    "sad": {"AU04", "AU01", "AU14", "AU15"},
    "fear": {"AU01", "AU02", "AU04", "AU05", "AU07", "AU20", "AU26"},
    "doubt": {"AU25"},
    "contempt": {"AU12", "AU10", "AU15", "AU17"},
    "neutral": set(),       
}
