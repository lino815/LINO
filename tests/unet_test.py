import pytest

def test_import_sucess():
    """
    Check if folder structure allows loading the model
    """
    from moonrise_fm.lightning_unet import LitUnet

class TestClass:
    AnySeed = 42
    AnySize = 161
    AnyBatchsize = 4
    def test_model_InputOneChannel_OutputOneChannel(self):
        """
        Check one channel in -> one channel out
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        x = torch.randn((self.AnyBatchsize, 1, self.AnySize, self.AnySize))  # batchsize, channels, sizex, sizey
        model = LitUnet(n_channels=1, n_classes=1)

        # act
        preds = model(x)

        # assert
        assert preds.shape == x.shape

    def test_model_InputTwoChannels_OutputTwoChannels(self):
        """
        Check two channels in -> two channel out
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        x = torch.randn((self.AnyBatchsize, 2, self.AnySize, self.AnySize))  # batchsize, channels, sizex, sizey
        model = LitUnet(n_channels=2, n_classes=2)

        # act
        preds = model(x)

        # assert
        assert preds.shape == x.shape

    def test_model_InputOneChannel_OutputTwoChannels(self):
        """
        Check one channel in -> two channels out
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        x = torch.randn((self.AnyBatchsize, 1, self.AnySize, self.AnySize))  # batchsize, channels, sizex, sizey
        model = LitUnet(n_channels=1, n_classes=2)

        # act
        preds = model(x)

        # assert
        assert preds.shape == (self.AnyBatchsize, 2, self.AnySize, self.AnySize)

    def test_model_InputTwoChannels_OutputOneChannel(self):
        """
        Check two channels in -> one channel out
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        x = torch.randn((self.AnyBatchsize, 2, self.AnySize, self.AnySize))  # batchsize, channels, sizex, sizey
        model = LitUnet(n_channels=2, n_classes=1)

        # act
        preds = model(x)

        # assert
        assert preds.shape == (self.AnyBatchsize, 1, self.AnySize, self.AnySize)

    def test_model_InputOddImageSize_OutputOddImageSize(self):
        """
        Check if images with odd images pass
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        x = torch.randn((self.AnyBatchsize, 1, 101, 101))  # batchsize, channels, sizex, sizey
        model = LitUnet(n_channels=1, n_classes=1)

        # act
        preds = model(x)

        # assert
        assert preds.shape == (self.AnyBatchsize, 1, 101, 101)

    def test_model_InputEvenImageSize_OutputEvenImageSize(self):
        """
        Check if images with even images pass
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        x = torch.randn((self.AnyBatchsize, 1, 100, 100))  # batchsize, channels, sizex, sizey
        model = LitUnet(n_channels=1, n_classes=1)

        # act
        preds = model(x)

        # assert
        assert preds.shape == (self.AnyBatchsize, 1, 100, 100)

    def test_model_InputTooSmallImageSize_OutputRuntimeError(self):
        """
        Check minimal image size. Every block halves the image size, and we have the same number of up/down blocks.
        Thus, the minimal image size is 2^(#blocks)
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        x = torch.randn((self.AnyBatchsize, 1, 10, 10))  # batchsize, channels, sizex, sizey
        model = LitUnet(n_channels=1, n_classes=1)

        # act
        with pytest.raises(RuntimeError, match='Output size is too small') as exc_info:
            preds = model(x)

        # assert
        assert exc_info.type is RuntimeError

    def test_model_InputMinimalImageSize_OutputNoError(self):
        """
        Check minimal image size. Every block halves the image size, and we have the same number of up/down blocks.
        Thus, the minimal image size is 2^(#blocks)
        """
        # init
        import torch
        from moonrise_fm.lightning_unet import LitUnet
        torch.manual_seed(self.AnySeed)
        model = LitUnet(n_channels=1, n_classes=1)
        number_of_children = len(list(model.children()))
        # when the lowest size is 1x1 you have to start with an image size of 2 ** (no. downblocks)
        minimalSize = int(2**(number_of_children/2))
        x = torch.randn((self.AnyBatchsize, 1, minimalSize, minimalSize))  # batchsize, channels, sizex, sizey

        # act and assert
        try:
            preds = model(x)
        except RuntimeError as exc:
            assert False, f" minimal input of  raised an exception {exc}"