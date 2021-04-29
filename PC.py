import torch
from torch import nn
from utilities import sigmoid_d
from utilities import tanh_d

relu = torch.nn.ReLU()
mse = torch.nn.MSELoss(reduction='none')


class PC(nn.Module):

    def __init__(self, layer_sizes, weight_decay=.01, f_l_rate=.001, online=True, train_err_wts=False,
                 n_iter=5, bot_infer_rate=.1, top_infer_rate=.1, true_gradients=False, func=torch.tanh, func_d=tanh_d, div_err=False,
                 bp_train=False, pos_acts=True, bias=0):
        super().__init__()

        self.input_sz = layer_sizes[0]
        self.layer_szs = layer_sizes
        self.out_sz = layer_sizes[-1]
        self.num_layers = len(layer_sizes)
        self.num_iter = n_iter

        self.f_l_rate = f_l_rate
        self.bot_infer_rate = bot_infer_rate
        self.top_infer_rate = top_infer_rate

        self.forward_wts, self.error_wts = self.make_weights()
        self.f_optims, self.e_optims = self.make_optims()
        self.bp_optim = torch.optim.Adam(self.forward_wts.parameters(), lr=self.f_l_rate)

        self.f_decay = weight_decay * self.f_l_rate

        self.train_online = online
        self.true_grads = true_gradients
        self.train_err_wts = train_err_wts
        self.bp_train = bp_train

        self.div_err = div_err
        self.bias = bias

        self.func = func
        self.func_d = func_d
        self.pos_acts = pos_acts


    def make_weights(self):
        f = [nn.Linear(self.layer_szs[0], self.layer_szs[1], bias=False)]
        e = []

        for i in range(self.num_layers - 2):
            f.append(nn.Linear(self.layer_szs[i+1], self.layer_szs[i+2], bias=False))
            e.append(nn.Linear(self.layer_szs[i+2], self.layer_szs[i+1], bias=False))

        return nn.ModuleList(f), nn.ModuleList(e)


    def make_optims(self):

        f = [torch.optim.Adam(self.forward_wts[0].parameters(), lr=self.f_l_rate)]
        e = []

        for n in range(self.num_layers - 2):
            f.append(torch.optim.Adam(self.forward_wts[n + 1].parameters(), lr=self.f_l_rate))
            e.append(torch.optim.Adam(self.error_wts[n].parameters(), lr=self.f_l_rate))

        return f, e


    def initialize_values(self, x, activities, predictions, global_target):

        activities[0] = x.clone()
        for i in range(1, self.num_layers):
            predictions[i-1] = self.forward_wts[i-1](activities[i-1])
            activities[i] = self.func(predictions[i-1]) + self.bias
            if self.pos_acts:
                activities[i] = relu(activities[i])
        activities[-1] = global_target + self.bias


    def compute_errors(self, activities, predictions, errors):
        for l in range(self.num_layers-1):

            if self.div_err:
                errors[l] = torch.clamp(torch.sqrt((activities[l+1] + .001) / (self.func(predictions[l]) + self.bias + .001)), 0, 100)   #original
            else:
                if l == self.num_layers-2:
                    errors[l] = activities[l+1] - (torch.sigmoid(predictions[l]) + self.bias)
                else:
                    errors[l] = activities[l+1] - (self.func(predictions[l]) + self.bias)


    def get_delta(self, error, prediction, l):

        if self.div_err:
            #return -error * (1 - error) * (torch.square(self.func_d(prediction)) / self.func(prediction))
            return torch.log(error) * (self.func_d(prediction) / (self.func(prediction) + self.bias + .001))  # original
        else:
            if l == self.num_layers-2:
                return error * sigmoid_d(prediction)
            else:
                return error * self.func_d(prediction)



    def get_top_err(self, error, activity, prediction):

        if self.div_err:
            #return error * ((1 / activity) - (1 / self.func(prediction)))
            return torch.log(error) * (1 / (activity + .1))
        else:
            return error.clone()



    def inference_step(self, activities, errors, predictions):

        #Update hidden layer activities by taking gradient step over activities
        for l in range(1, self.num_layers-1):

            delta = self.get_delta(errors[l], predictions[l], l)

            # Bottom up error
            if self.true_grads:
                botm_err = delta.matmul(self.forward_wts[l].weight)
            else:
                botm_err = self.error_wts[l-1](delta)

            top_err = self.get_top_err(errors[l-1], activities[l], predictions[l-1])

            activities[l] += self.bot_infer_rate * botm_err - self.top_infer_rate * top_err

            if self.pos_acts:
                activities[l] = relu(activities[l])


    def predict(self, predictions, activities):
        for i in range(0, self.num_layers-1):
            predictions[i] = self.forward_wts[i](activities[i])


    def inference(self, activities, predictions, errors):

        for i in range(self.num_iter):

            #Compute errors
            self.compute_errors(activities, predictions, errors)

            #Update activities
            old_activities = [activities[x].clone() for x in range(len(activities))]
            self.inference_step(activities, errors, predictions)

            # Update weights
            if self.train_online:
                self.train_step(old_activities, activities, predictions, errors)

            #Update predictions
            self.predict(predictions, activities)




    #This function used if we do not train forward weights online.
    def train_step(self, old_activities, activities, predictions, errors):

        #self.predict(predictions, activities)
        self.compute_errors(activities, predictions, errors)


        for w_num in range(self.num_layers - 1):

            # Compute delta at each level and gradient of delta w.r.t. weights
            #delta = errors[w_num] * self.func_d(predictions[w_num])
            delta = self.get_delta(errors[w_num], predictions[w_num], w_num)
            dw = delta.t().matmul(old_activities[w_num])

            # Zero out gradients, decay, and perform gradient step
            self.forward_wts[w_num].weight.grad = torch.zeros_like(self.forward_wts[w_num].weight)
            self.forward_wts[w_num].weight.grad -= dw
            self.forward_wts[w_num].weight -= self.f_decay * self.forward_wts[w_num].weight
            self.f_optims[w_num].step()

            ##Zero out gradients, decay, and perform gradient step for error weights
            if w_num > 0 and self.train_err_wts:
                self.error_wts[w_num - 1].weight.grad = torch.zeros_like(self.error_wts[w_num - 1].weight)
                self.error_wts[w_num - 1].weight.grad -= dw.t()
                self.error_wts[w_num - 1].weight -= self.f_decay * self.error_wts[w_num - 1].weight
                self.e_optims[w_num - 1].step()


    def train_bp(self, img, target):

        x = self.func(self.forward_wts[0](img.detach()))
        for i in range(1, self.num_layers-1):
            x = self.func(self.forward_wts[i](x))

        loss = torch.mean(mse(x, target.detach()).sum(1))

        self.bp_optim.zero_grad()
        loss.backward()
        self.bp_optim.step()


    def train_network(self, img, activities, predictions, errors, target):

        self.initialize_values(img, activities, predictions, target)
        self.inference(activities, predictions, errors)

        if not self.train_online:
            self.train_step(activities.copy(), activities, predictions, errors)
