import torch
from train import build_model


def main():

    model = build_model('resnet34', 'square_crop', pretrained=True)
    print(model)

    input1 = torch.zeros(32, 1, 137, 236)
    input2 = torch.zeros(32, 1, 128, 128)
    out = model.forward((input1, input2))
    print(out.shape)


if __name__ == '__main__':
    main()
