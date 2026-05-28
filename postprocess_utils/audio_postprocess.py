import torch


def inverse_stft(log_magnitude, phase, n_fft=512, hop_length=256):
    if log_magnitude.ndim == 4:
        log_magnitude = log_magnitude.squeeze(1).float()
    if phase.ndim == 4:
        phase = phase.squeeze(1).float()


    magnitude = (10 ** log_magnitude) - 1

    pad_mag = torch.cat([magnitude, torch.zeros(magnitude.shape[0], 1, magnitude.shape[-1]).to(magnitude.device)], dim=1)
    pad_phase = torch.cat([phase, torch.zeros(phase.shape[0], 1, phase.shape[-1]).to(phase.device)], dim=1)

    stft_matrix = pad_mag * torch.exp(1j * pad_phase)
    window = torch.hann_window(n_fft).to(stft_matrix.device)

    restored_audio = torch.istft(
        stft_matrix, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=window
    )
    return restored_audio