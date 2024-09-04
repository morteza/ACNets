"""Keras3 automatically enables MPS if available. This test implements a
   hack to manually disable MPS."""

def test_change_keras_torch_device():
    import os
    os.environ['KERAS_BACKEND'] = 'torch'
    import keras
    assert keras.backend.backend() == 'torch'
    assert keras.src.backend.torch.core.get_device() == 'mps'

    keras.src.backend.common.global_state.set_global_attribute('torch_device', 'cpu')
    assert keras.src.backend.torch.core.get_device() == 'cpu'

def test_mps_availability():
    import torch
    assert torch.backends.mps.is_available()
