import numpy as np
import torch
from omegaconf import OmegaConf

from protocouncil.import_prostrl import PREFIX, PROMPT_NAMES, convert
from protocouncil.models.detector import Detector
from protocouncil.models.encoders import MedSAMEncoder
from tests.test_detector_and_data import make_manifest


def test_prostrl_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    det = Detector(MedSAMEncoder(), num_classes=2).eval()
    back = {v: k for k, v in PROMPT_NAMES.items()}
    sd = {}
    for k, v in det.state_dict().items():  # Prost-RL key layout (policy/value modules are ignored by the import)
        if k.startswith('encoder.encoder.'):
            sd[PREFIX + 'medsam_model.image_encoder.' + k[len('encoder.encoder.'):]] = v
        elif k.startswith(('prompt_encoder.', 'mask_decoder.')):
            sd[PREFIX + 'medsam_model.' + k] = v
        elif k.startswith('prompt_mlps.'):
            name, rest = k[len('prompt_mlps.'):].split('.', 1)
            sd[f'{PREFIX}floating_point_prompt_modules.{back[name]}.{rest}'] = v
        else:
            sd[PREFIX + k] = v
    sd['model.policy.attention_head.weight'] = torch.zeros(1)
    torch.save(dict(model=sd, args=dict(name='x', data=dict(image_size=256, n_folds=5))), tmp_path / 'ck.pth')
    new, _, n = convert(tmp_path / 'ck.pth')
    x, pr = torch.randn(1, 3, 256, 256), torch.rand(1, 5)
    a, b = det(x, pr, return_tokens=True), new.eval()(x, pr, return_tokens=True)
    assert n == 2 and all(torch.equal(a[k], b[k]) for k in a)


def test_random_anchor_bank_is_deterministic(tmp_path):
    from research_detector.anchors import random_bank, resolve_bank
    a, b = random_bank(), random_bank()
    assert np.array_equal(a['anchors'], b['anchors']) and a['anchors'].shape == (60, 256)
    assert np.allclose(np.linalg.norm(a['anchors'], axis=1), 1, atol=1e-6)
    assert np.bincount(a['grade_labels']).tolist() == [10] * 6 and np.bincount(a['state_labels']).tolist() == [10, 20, 30]
    assert resolve_bank('random', tmp_path).exists()


def test_research_batches_have_the_expected_fields(tmp_path):
    from research_detector.data import PROMPT_FIELDS, get_dataloaders
    args = OmegaConf.create(dict(manifest=str(make_manifest(tmp_path)), fold=0, n_folds=5, batch_size=2, num_workers=0,
                                 exclude_benign_cores_from_positive_patients=True, augmentations='none', image_size=256,
                                 mask_size=64, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    b = next(iter(get_dataloaders(args)['train']))
    for key in ('bmode', 'prostate_mask', 'needle_mask', 'label', 'grade_group', 'clinically_significant', 'involvement',
                'involvement_valid', 'cores_positive_for_patient', 'core_id', 'patient_id', 'center') + PROMPT_FIELDS:
        assert key in b, key
    assert b['bmode'].shape[1:] == (3, 256, 256) and b['prostate_mask'].shape[1:] == (1, 64, 64)


def test_prostrl_fold_must_match(tmp_path):
    import pytest
    from protocouncil.import_prostrl import check_split
    check_split(dict(data=dict(fold=2, n_folds=5)), 2, 5, 'ck')
    for bad in (dict(fold=1, n_folds=5), dict(fold=1), dict(n_folds=10), dict(fold=0, n_folds=10)):
        with pytest.raises(ValueError):
            check_split(dict(data=bad), 0, 5, 'ck')
    check_split(dict(data=dict(fold=0)), 0, 5, 'ck')  # partial metadata that agrees: warning only
    check_split({}, 0, 5, 'ck')


def test_prostrl_export_keeps_the_requested_fold_count(tmp_path):
    from protocouncil.import_prostrl import save
    from tests.test_detector_and_data import TinyEncoder
    save(Detector(TinyEncoder(), num_classes=2), 2, dict(data={}), 3, 10, tmp_path / 'x.pth')
    ck = torch.load(tmp_path / 'x.pth', weights_only=False)
    assert (ck['fold'], ck['n_folds']) == (3, 10)


def test_decoder_init_only_skips_the_class_output_layer():
    import pytest
    from tests.test_detector_and_data import TinyEncoder
    src = Detector(TinyEncoder(), num_classes=2).state_dict()
    dst = Detector(TinyEncoder(), num_classes=3)
    assert dst.load_decoders_from(src) == ['class_decoder.output_cls_mlps.0.layers.2.bias',
                                           'class_decoder.output_cls_mlps.0.layers.2.weight']
    broken = {k: v for k, v in src.items() if 'output_cls_mlps.0.layers.1' not in k}
    with pytest.raises(AssertionError):
        dst.load_decoders_from(broken)
