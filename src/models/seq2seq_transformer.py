import torch
import torch.nn as nn
import metrics


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, device, emb_size, vocab_size, transformer_params, scheduler_params):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device

    # TODO: Реализуйте конструктор seq2seq трансформера - матрица эмбеддингов, позиционные эмбеддинги, encoder/decoder трансформер, vocab projection head
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.model = nn.Transformer(**transformer_params)
        self.emb_size = emb_size
        self.decoder = nn.Linear(emb_size, vocab_size)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)

    def forward(self, input_tensor: torch.Tensor):
        # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
        pass

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch
        (_, output) = self.forward(input_tensor, target_tensor)
        target = target_tensor.reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        pass

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
