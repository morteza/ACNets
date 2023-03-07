import os


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id ([N]VGPxx[NEW])
    seqitem: run number during scanning
    subindex: sub index within group
    session: scan index for longitudinal acq (1, 2)
    """

    # localizer = create_key('{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-localizer_bold')
    t1w = create_key('{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_T1w')
    fmap_mag = create_key('{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_run-{item:02d}_magnitude')
    fmap_phase = create_key('{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_run-{item:02d}_phasediff')
    func = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-attention_run-{item:02d}_bold')

    info = {
        # localizer: [],
        t1w: [],
        fmap_phase: [],
        fmap_mag: [],
        func: []
    }

    # last_run = len(seqinfo)

    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_id
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """
        # if 'localizer' in s.protocol_name:
        #     info[localizer].append(s.series_id)
        if 'MPRAGE T1' in s.protocol_name:
            info[t1w].append(s.series_id)
        if ('field_mapping' in s.protocol_name) and ('M' in s.image_type) and (s.dim3 == 72):
            info[fmap_mag].append(s.series_id)
        if ('field_mapping' in s.protocol_name) and ('P' in s.image_type) and (s.dim3 == 36):
            info[fmap_phase].append(s.series_id)
        if 'ep2d_bold' in s.protocol_name:
            info[func].append(s.series_id)
    return info


def to_bids_session_id(julia_session_id):
    mapping = {
        'Attention1': 'A1',
        'Attention2': 'A2'
    }
    return mapping[julia_session_id]


def to_bids_subject_id(julia_subject_id):
    # TODO NVGP/VGP -> NVGP/AVGP
    return julia_subject_id


def filter_files(fl):
    if any(p in os.path.basename(fl) for p in ['.nii.gz', '.DS_Store']):
        return False
    return True


# def infotoids(seqinfos, outdir):
#     seqinfos = list(seqinfos)
#     subject = to_bids_subject_id(seqinfos[0].patient_id)
#     return {
#         'subject': subject,
#         'session': seqinfos[0].date}
