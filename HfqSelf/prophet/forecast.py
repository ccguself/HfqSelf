from prophet.model import FCN, FCN_1
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class forecastNet:
    def __init__(self, args, save_file):

        self.args = args
        self.save_file = save_file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.model_type == 0:
            self.model = FCN(
                args.input_dim,
                args.hidden_dim,
                args.output_dim,
                args.in_seq_length,
                args.out_seq_length,
                args.conv,
                self.device,
            )
        if args.model_type == 1:
            self.model = FCN_1(
                args.input_dim,
                args.hidden_dim,
                args.output_dim,
                args.in_seq_length,
                args.out_seq_length,
                self.device,
            )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate
        )

    @staticmethod
    def format_input(input):
        """
        Format the input array by combining the time and input dimension of the input for feeding into ForecastNet.
        That is: reshape from [in_seq_length, batch_size, input_dim] to [batch_size, in_seq_length * input_dim]
        :param input: Input tensor with shape [in_seq_length, batch_size, input_dim]
        :return: input tensor reshaped to [batch_size, in_seq_length * input_dim]
        """
        in_seq_length, batch_size, input_dim = input.shape
        input_reshaped = input.permute(1, 0, 2)
        input_reshaped = torch.reshape(input_reshaped, (batch_size, -1))
        return input_reshaped

    def evaluate(self, test_x):

        self.model.eval()
        checkpoint = torch.load(self.save_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        with torch.no_grad():
            if type(test_x) is np.ndarray:
                test_x = (
                    torch.from_numpy(test_x).type(torch.FloatTensor).to(self.device)
                )
            test_x = self.format_input(test_x)
            empty_y = torch.empty(
                (
                    sum([self.args.out_seq_length // c for c in self.args.conv]),
                    test_x.shape[1],
                    self.args.output_dim,
                )
            )
            y_pred = self.model(test_x, empty_y, is_training=False)
        return y_pred.cpu().numpy()

    def train(
        self,
        train_x,
        train_y,
        validation_x=None,
        validation_y=None,
        restore_session=False,
        result_prt=True,
    ):
        """
        :param train_x: Input training data in the form [input_seq_length, batch_size, input_dim]
        :param train_y: Target training data in the form [output_seq_length, batch_size, output_dim]
        :return: training_costs: a list of training costs over the set of epochs
        :return: validation_costs: a list of validation costs over the set of epochs
        """

        # Convert numpy arrays to Torch tensors
        if type(train_x) is np.ndarray:
            train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        if type(train_y) is np.ndarray:
            train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
        if type(validation_x) is np.ndarray:
            validation_x = (
                torch.from_numpy(validation_x).type(torch.FloatTensor).to(self.device)
            )
        if type(validation_y) is np.ndarray:
            validation_y = (
                torch.from_numpy(validation_y).type(torch.FloatTensor).to(self.device)
            )

        train_x = self.format_input(train_x)
        validation_x = self.format_input(validation_x)

        if restore_session:
            checkpoint = torch.load(self.save_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        n_samples = train_x.shape[0]
        training_costs = []
        validation_costs = []
        early_stop = 0
        self.model.train()

        for epoch in range(self.args.n_epochs):

            t_start = time.time()
            print("Epoch: %i of %i" % (epoch + 1, self.args.n_epochs))
            batch_cost = []

            for sample in range(0, n_samples, self.args.batch_size):
                input = train_x[sample : sample + self.args.batch_size, :].to(
                    self.device
                )
                target = train_y[:, sample : sample + self.args.batch_size, :].to(
                    self.device
                )
                self.optimizer.zero_grad()

                outputs = self.model(input, target, is_training=True)

                loss = F.mse_loss(
                    input=outputs[:, :, : self.args.prd_n],
                    target=target[:, :, : self.args.prd_n],
                )
                batch_cost.append(loss.item())
                loss.backward()
                self.optimizer.step()
            epoch_cost = np.mean(batch_cost)
            training_costs.append(epoch_cost)

            if validation_x is not None:
                self.model.eval()
                with torch.no_grad():
                    y_valid = self.model(validation_x, validation_y, is_training=False)
                    if result_prt:
                        self.class_prt(validation_y, y_valid, self.args.t_unit)
                    #                     self.next_prt(validation_y,y_valid,self.args.t_unit)
                    loss = F.mse_loss(
                        input=y_valid[:, :, : self.args.prd_n],
                        target=validation_y[:, :, : self.args.prd_n],
                    )
                    validation_costs.append(loss.item())
                self.model.train()

            print("Average epoch training cost: ", epoch_cost)
            if validation_x is not None:
                print("Average validation cost:     ", validation_costs[-1])
            print("Epoch time:                   %f seconds" % (time.time() - t_start))
            print(
                "Estimated time to complete:   %.2f minutes, (%.2f seconds)"
                % (
                    (self.args.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
                    (self.args.n_epochs - epoch - 1) * (time.time() - t_start),
                )
            )

            # Save a model checkpoint
            best_result = False
            if validation_x is None:
                if training_costs[-1] == min(training_costs):
                    best_result = True
            else:
                if validation_costs[-1] == min(validation_costs):
                    best_result = True
            if best_result:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.save_file,
                )
                print("Model saved in path: %s" % self.save_file)
            else:
                early_stop += 1
                if early_stop > 3:
                    print("EARLY STOP")
                    break

        return training_costs, validation_costs

    @staticmethod
    def class_prt(y, y_hat, t):
        pl = [0, 1, 2, 3, 4, 5]
        sigmoid = lambda x: 1 / (1 + np.exp(-x / 5))
        t = sigmoid(t)
        pl = [sigmoid(0), sigmoid(1), sigmoid(2), sigmoid(3), sigmoid(4), sigmoid(5)]
        pc, pf, pn = (
            [0 for _ in range(len(pl))],
            [0 for _ in range(len(pl))],
            [0 for _ in range(len(pl))],
        )
        nc, nf, nn = (
            [0 for _ in range(len(pl))],
            [0 for _ in range(len(pl))],
            [0 for _ in range(len(pl))],
        )
        p3c, p3f = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        n3c, n3f = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        for i in range(1):
            for j in range(y.shape[1]):
                for k in range(len(pl)):
                    if y[i][j][0] >= pl[k] and y_hat[i][j][0] >= pl[k]:
                        pc[k] += 1
                    if y[i][j][0] < pl[k] and y_hat[i][j][0] >= pl[k]:
                        pf[k] += 1
                    if y[i][j][0] >= pl[k] and y_hat[i][j][0] < pl[k]:
                        pn[k] += 1
                    if y[i][j][0] >= t and y_hat[i][j][0] >= pl[k]:
                        p3c[k] += 1
                    if y[i][j][0] < t and y_hat[i][j][0] >= pl[k]:
                        p3f[k] += 1

                    if y[i][j][1] >= pl[k] and y_hat[i][j][1] >= pl[k]:
                        nc[k] += 1
                    if y[i][j][1] < pl[k] and y_hat[i][j][1] >= pl[k]:
                        nf[k] += 1
                    if y[i][j][1] >= pl[k] and y_hat[i][j][1] < pl[k]:
                        nn[k] += 1
                    if y[i][j][1] >= t and y_hat[i][j][1] >= pl[k]:
                        n3c[k] += 1
                    if y[i][j][1] < t and y_hat[i][j][1] >= pl[k]:
                        n3f[k] += 1

        print(pc)
        print(pf)
        print(pn)
        print("------")
        print(nc)
        print(nf)
        print(nn)
        print("-----")
        print(p3c)
        print(p3f)
        print(
            [
                p3c[i] / (p3c[i] + p3f[i]) if p3c[i] + p3f[i] > 0 else 0
                for i in range(len(p3c))
            ]
        )
        print(n3c)
        print(n3f)
        print(
            [
                n3c[i] / (n3c[i] + n3f[i]) if n3c[i] + n3f[i] > 0 else 0
                for i in range(len(p3c))
            ]
        )
        return

    @staticmethod
    def next_prt(y, y_h, t):
        pl = [0, 1, 2, 3, 4, 5]
        sigmoid = lambda x: 1 / (1 + np.exp(-x / 5))
        pl = [sigmoid(0), sigmoid(1), sigmoid(2), sigmoid(3), sigmoid(4), sigmoid(5)]
        pc, pw = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        nc, nw = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        for j in range(y.shape[1]):
            for k in range(len(pl)):
                if (
                    y_h[0][j][0] >= pl[k]
                    and y_h[10][j][0] >= pl[k]
                    and y_h[15][j][0] >= pl[k]
                    and y_h[18][j][0] >= pl[k]
                ):
                    if y[0][j][0] >= t:
                        pc[k] += 1
                    else:
                        pw[k] += 1
                if (
                    y_h[0][j][1] >= pl[k]
                    and y_h[10][j][1] >= pl[k]
                    and y_h[15][j][1] >= pl[k]
                    and y_h[18][j][1] >= pl[k]
                ):
                    if y[0][j][1] >= t:
                        nc[k] += 1
                    else:
                        nw[k] += 1
        print("Union Predict:")
        print(pc)
        print(pw)
        print(
            [
                pc[i] / (pc[i] + pw[i]) if pc[i] + pw[i] > 0 else None
                for i in range(len(pc))
            ]
        )
        print(nc)
        print(nw)
        print(
            [
                nc[i] / (nc[i] + nw[i]) if nc[i] + nw[i] > 0 else None
                for i in range(len(pc))
            ]
        )

        pc, pw = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        nc, nw = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        for j in range(y.shape[1]):
            for k in range(len(pl)):
                if y_h[18][j][0] >= pl[k]:
                    if y[18][j][0] >= t + 1:
                        pc[k] += 1
                    else:
                        pw[k] += 1
                if y_h[18][j][1] >= pl[k]:
                    if y[18][j][1] >= t + 1:
                        nc[k] += 1
                    else:
                        nw[k] += 1
        print("Next_5m >=", t + 1, " Predict:")
        print(pc)
        print(pw)
        print(
            [
                pc[i] / (pc[i] + pw[i]) if pc[i] + pw[i] > 0 else None
                for i in range(len(pc))
            ]
        )
        print(nc)
        print(nw)
        print(
            [
                nc[i] / (nc[i] + nw[i]) if nc[i] + nw[i] > 0 else None
                for i in range(len(pc))
            ]
        )

        pc, pw = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        nc, nw = [0 for _ in range(len(pl))], [0 for _ in range(len(pl))]
        for j in range(y.shape[1]):
            for k in range(len(pl)):
                if y_h[0][j][0] >= pl[k]:
                    if y[0][j][0] >= t:
                        pc[k] += 1
                    else:
                        pw[k] += 1
                if y_h[0][j][1] >= pl[k]:
                    if y[0][j][1] >= t:
                        nc[k] += 1
                    else:
                        nw[k] += 1
        print("Next_1m Predict:")
        print(pc)
        print(pw)
        print(
            [
                pc[i] / (pc[i] + pw[i]) if pc[i] + pw[i] > 0 else None
                for i in range(len(pc))
            ]
        )
        print(nc)
        print(nw)
        print(
            [
                nc[i] / (nc[i] + nw[i]) if nc[i] + nw[i] > 0 else None
                for i in range(len(pc))
            ]
        )
