# OTDD map

## 06/03/2022

modify it to OTDD distance.

## 06/05/2022

- [ ] need to compare with that swiss roll. In that example, the Gaussian appro may become a problem!!!

- [ ] also better compare the inbalanced classes

## 06/06/2022

- [x] train classifier singly: finished 9 components
- [x] modify discriminator structure to have fourier features.
- [x] train wgan singly: finished 9 components

For gmm9, 64 neurons, 5 layrers classfier is enough to do the task. However, inner_iter=2, it suddenly diverged. inner_iteration cannot be too large or too small, inner_iter=4 or 1 is the most stable now. It seems it totally depends on the random seeds.

## 06/09/2022

One interesting observation is if only keeping feature loss and discriminator loss, it's able to learn, if including the label loss, it disrupts. Even I set that coeff to be 0.1 and detach the feat input of classifier, it still spoils the pushforward badly.

coeff_feat cannot be too large e.g. 10.0. It cannot dominate the disciminator loss.

coeff_feat=1.0, coeff_label=0 works in the half, but later diverge, coeff_feat=1.0, coeff_label=1e-8 nearly works in the half, but later diverge

- [later] also show the monge map with only features

- [x] when their method can boil down to our method. only when the label correspondence aligns with the feature similarity, we are the same as theirs, such as MNIST-USPS example.

## 06/10/2022

Change the strategy: 1) calculate the classifier in advance based on target domain. 2) add fourier feature to the classifier if it's not working well. 3) cannot detach the features.

- [x] verify my calculation of w distance is correct. It was wrong! Need to sort the labels.

## 06/11/2022

- [think] consider about the clustering, also need to do it at the end of each epoch.

- [no_effect] consider about detach vs non detach.

- [x] put label as the input of the classifer.

## 06/13/2022

- [x] (even worse??) try to cancel the label input to nn

- [later] try to freeze classifier at the beginning of training.

- [x] pretrain feature map to be identical map.

- [ ] think about loss. whether it is saddle point or not.

- [x] (visualization) having pushforward and target as well.

- [x] use conditional GAN discriminator structure.

## 06/16/2022

- [x] make the W2 table not dependent on the dataset.

- [ ] can use multi-scale embedding as pretraining. 1) learn from scratch, 2) fine-tuining. 3) zero-shot.

### 6/20/2022

Currently, I'm using the Wasserstein 2 distance on the feature space (FID) and the feature cost on the original space.

### 6/22/2022

cross entropy loss is nll loss + log + softmax.

- [x] maybe discriminator is too complex? Maybe no penalty? Switched to #feature=128.

- [x] Maybe I should try using just anti-identity matrix.

- [x] check what code did I use before. Oldest monge map paper uses MLP and final layer tanh.

### 7/3/2022

- [x] try to fix classifier during training, and not using conditioned discriminator.

Need to modify data to three channels.

### 7/4/2022

Now anti-principle: 1) W2 distance didn't use correct one 2) [okay] classifier didn't use label input 3) [okay] didn't make classifier trainable during main training.

Note that after adding correct W distance, the map may not preserve class

- [x] put source ordered by classes.

- [x] add correct w distance.

- [x] put label loss and feat loss the same level as dis loss.

- [x] try discriminator without label input (still mode collapse)

- [no_need] try discriminator with another position of projection

- [x] maybe add some noise to the classifier. (not working, classifier accuracy also has no difference. Maybe I didn't add strong connection of noise in classifier, previously with noise, classifier can have accuracy gain)

- [x] to recover to bunne: cross entropy loss (works better?)

- [x] to recover to bunne: without pretraining + cross entropy loss (not good)

- [ ] the last choice is to add spectral normalization

The mode collapse is not related to inner iteration nor magnitude of feature loss. It's related to feature loss, if I set feature loss the same order as discriminator loss, it can avoid mode collapse, but it won't preserve the labels.

### 7/6/2022

- [x] add the classifier dump there.

- [no] add FID

I find if the feature loss / label loss ratio is too large like 1e4, then it would have mode collapse.

### 7/13/2022

I added the correct w distances.

Now I use shifted w distance for all of them. When I plot the curve, I need to put them back.

### 7/14/2022

talk about geodesic, how to combine two maps, overfitting issue.

1. Detect whether my otdd map is doing the correct thing.

- [x] use classifer histogram for pushforwad samples. Important: plot the distribution of target ground truth as well.

- [no_need] doing the same thing as transfer learning

- [x] use feature for both or not both.

feature loss: origin, label loss: origin, 3 not showing up.

feature loss: origin, label loss: None, 3 not showing up.

feature loss: origin, label loss: feature, 3 showing up.

feature loss: feature, label loss: feature, diverge...

2. See the combination of maps, whether it works?? done

3. Still calculate the generalized geodesic.

I find in the ipynb, I wasn't using data transformer!!!!! That's why loss is so easy to rocket up!!

### 7/25/2022

USPS dataset has lots of 0 and 1, others are roughly the same. 3 is more than 5.

MNIST classes are balanced.

### 7/27/2022

Get results of the Gaussian mixture.

- [not_now] point-cloud data?? each sample is a point cloud.

The final goal of our project??

- [ ] data augmentation for imbalanced data??

Now:

- [x] draw the process of geodesic during training.

- [x] do a unrealistic task: MNIST and USPS, we have enough samples. (use 0-nonzero can make sure we have balanced usps pushforward datasets.)

- [x] realistic: if we have FashionMNIST and EMNIST, then fine tune it on others.

I find my label loss is larger with the same resnet18, the reason is now I'm using train mode instead of eval mode. lightning automatically makes it in train mode

- [x] need to find a correct way of saving the best model for MNIST-USPS. I think FID is good enough/add ema. (later, just rerun it) need to save the path more smartly. I finally added ema instead of using best FID.

### 7/28/2022

I figure out this ema is 1) you only need to save the ema, not the map parameters. 2) you save the model parameter under the `with pl_module.ema_map.average_parameters():` context.

Now FashionMNIST -> EMNIST: not working well. Maybe it's because they're too far away.

- [x] eval mode for classifier

- [x] diverse -> less diverse

- [not] one last step: drop some class.

### 8/3/2022

- [x] Implementing the train last two layer code using interpolations. 1) interpolating between unequal number of class 2) dump the last layer and retrain classifier??

A problem in our method: even we assume label size can be different, the feature size has to be the same.

Meeting for Thursday: 1) David's idea of interpolating? 2) transfer learning results. 2) document the connection between map and barycentric mapping.

### 8/4/2022

Train on interpolation between EMNIST and FMNIST. Results of fine-tuning on

- [x] Now USPS is doing weird thing, it should increase straight. I should compare with training on a ground truth USPS.

- [x] plot the few shot samples for sanity check

Training over interpolations need about 17 minutes for EMNIST to train one epoch for each time.

One reason of USPS bad result maybe those distances are calculated on pixel space? Maybe need to recalculate them based on feature sapce?

- [x] check the feature space: whether USPS is closer to EMNIST or FMNIST? on resnet18 and !!debiased. Calculate all those numbers.

Embedded OTDD(FMNIST,USPS) = 174.61
Embedded OTDD(EMNIST,USPS) = 95.23

Consistent with my curve

- [x] try cross entropy using soft label to train (retrain everything)

- [x] average over several random seeds, save the numpy arrays, raw data. (retrain everything, and repeat many times) First try only randomize the fine-tuning, then try randomize the training as well.

- [ ] document the connection between map and barycentric mapping

### 8/5/2022

begin to do generalized geodesic. 1) clustering 2) map with clustering

I've done something about that. I find if using coeff_label=1.0, the mapping has mode collapse, and visually not that realistic. coeff_label=0.1 is better.

Some problem of existing results:

1. MNIST <-> USPS accuracy increase, we use resnet18 with train mode when calculating label distance

2. EMNIST <-> FMNIST transfer learning, we use resnet18 with train mode when calculating label distance

I find a big problem: In mcCann's interpolation, I was using mixed feature to get pushforward label, I should only use pushforward feature!!

### 8/7/2022

I find pre-training the EMNIST classifier longer really decreases the performance of real training process, making the generator less stable.

- [x] use pixel space for everything.

- [x] use knn data for fine-tuning: not improving.

- [x] use more data for kmnist for mccann.

Previous methods are only heuristic. Create datasets -> having map is good. talk some monge map methods for

If linear evaluation, you'd better use eval mode for batch normalization.
And better do it in two steps: 1) linear evaluation on last layer first, 2) then train features.

- [x] see the accuracy converge in ternary plot?? in 100 epochs.

- [maybe_not] replace all pushforward end point by the ground truth training datasets.

- [x] ternary/quatery: test on emnist or usps or kmnist

- [x] mccann: test on kmnist and others, interpolate between mnist and usps.

- [x] plot the difference of finetuning and training from scratch, xaxis is the number of shots. do this for mcCann, would be cheaper.

### 8/10/2022

- [x] add config to those script files!!

- [half] 1. Optimization over weights. (more expensive) 2) use weights of euclidean distance.

- [x] otdd distance ternary plot 1) use euclidean space closed-form 2) use otdd on soft label??

Do we insist on fint-tuning only the last layer?? This sabotage the transfer learning accuracy a lot.

### 8/11/2022

- [not_now] add sanity check of train scratch.

- [x] add randomness in the few-shot knn data, add fixed random seeds in scripts.

- [x] make sure to be consistent with otdd solver. 1) training w2 matrix, use 3 channel 32 dimension (now it's 1 channel) -> calculate use otdd and then extract 2) coefficients are different. Dump it in a common folder.

- [x] otdd distance ternary plot with pushforward

- [not_now] need to implement sampler instead of epochs.

- [??] make dump interpolation and loader is faster.

- [added_early_stopping] use more flexible #epoch in otdd map training.

1. train otdd map: 2~4h (need be randomized for gen geodesic) 2) train classifier: random 5 seeds + 10 interpolation: 5h, 3) fine-tune: trivial: 10 minutes

downstream tasks/plotting: otdd ternary plot, ablation study of #shots.

Compare with synthetic data generation methods: 1) adapted mixup: our #data in interpolation = #data in source, mixup: #interpolaion data: #source data \* #target data. Use the same of #samples. 2) barycentric mapping on ipad. 3) train from scratch.

What do we compare? Test accuracy

### 8/12/2022

Whether it's squared w2 distance or not?? squared

randomness in accuracy: fine-tuning << training << few-shot data (affect otdd map)

randomness in otdd: maxsamples, sub-sampling.

### 8/15/2022

Where do we need to solve OTDD:

1. OTDD as a target value baseline for OTDD map, will be used in OTDD ternary plot as well. 1) between two full datasets 2) between one few-shot, one full. There are (2\*#seed+1) \* C_5^2 choices.

2. OTDD between pushforward and target as a quantity criteria.

### 8/18/2022

The Camelyon17 dataset
https://wilds.stanford.edu/datasets/

- [x] add randomness to gen. geodesic fine-tuning, see whether it has any improvement. If not, just use one random seed.

- [x] implement the mixup.

- [x] implement the barycentric mapping with otdd cost. Difference with network: 1) regularization bias: the pushforward labels could already be very soft smoothed, non-singular. 2) the number of data used in calculating coupling is generally not from full data. I kind of worry this would be exactly the same as ours. Maybe we should just sell: in small dataset, use barycentric mapping, in large dataset, use ours. \*NIST are 3072, they can handle? right?

<!-- add randomness into mccann interpolation and redraw bandwidth curves. Only MNIST <-> USPS, and fine-tune on others. -->

- [x] Now the feature loss is too large, check whether it makes sense.

- [wait_to_see] add the target subsets OTDD as lower bound.

### 8/21/2022

Now, OTDD and accuracy don't match very well. 1) to add random seeds (randomness work for KMNIST); 2) try different combinations. (tried someone, maybe work?) 3) try only fine-tuning last layer?

### 8/25/2022

- [later] fix the epoch of trained otdd map.

- [ ] flip the color of ternary plots.

- [ ] fix the range of those plots.

Writing:

- [ ] draw diagrams for the mcCann interpolation/gen. geodesic.

- [ ] I can reuse the datasets of generalized geodesic interpolated dataset.

### 8/30/2022

All of those: we cannot compare with 1) mixup, 2) barycentric mapping, 3) training from scratch 4) 1-KNN.

- [x] Generate data for insufficient target dataset, and use it to train the classifier. Use EMNIST -> MNIST-M 1) check whether KNN is good at this accuracy. It's only 30%~33%. train from scratch: ~54% 2) see the result of barycentric mapping, and mixup. This example doesn't make much sense, mixup would definitely be better than barycentric mapping because barycentric mapping is a subset of mixup. Maybe OTDD can be better because OTDD may create something (or roughly similar because of mode collapsing). 3) OTDD map won't work because the classifier result on boosted dataset won't be better than the pre-classifier, which is essentially trained from scratch ~54% or trained on mixup/knn, not better.

I think this one would determine whether we need to keep the neural solver.

<!-- d1=3, d2=2, d3=2
a*[0,0,1,0,0,0,0] + b*[0,0,0,1,0,0,0] + c\*[0,0,0,0,0,1,0], a+b+c=1. -->

### 9/5/2022

I checked the standard deviation, there is no pattern, who is larger, who is smaller.

I find the pretrained classifier is only 90~93% accuracy for FMNIST and KMNIST, but I don't find a correlation between large gap and pretrained accuracy.
The gap: MNIST > KMNIST > EMNIST > USPS > FMNIST.
But accuracy order: MNIST > USPS > KMNIST = EMNIST > FMNIST. Only USPS is very weird.
Better learnt classifier has larger gap????? It seems so weird, but it should mean label is not important? The important difference is image quality/diversity? So, maybe when diversity is already not enough in the original dataset, we can be nearly the same, like PCAM???

I checked the results of traning from scratch. It's worse than OTDD map.

I plan to give up that experiment: train on A - B, test on C. I think that's not useful in reality. The curve looks bad and even if I involve more randomness, it's not that random.

### 9/6/2022

- [ ] add the pcam dataset results. Do mixup first, then barycentric mapping, then otdd map. I have issue when downloading the dataset. It's too large, and it's split into different hospitals. The split scheme is putting all training datasets together????

### 9/16/2022

- [x] make it 3d. https://github.com/sbyebss/Scalable-Wasserstein-Barycenter/blob/master/optimal_transport_modules/plot_utils.py

### 9/22/2022

- [x] should use neural + knn or barycentric mapping for the ternary plots because not accurate classifier can introduce some bias.

I tried increasing # of fine-tuning in 20-shot test datasets, trying to beat knn, but it cannot. I've also tried 5-shot, KMNIST there is no hope to beat 1-NN.

- [ ] use 5 shots. fine-tune longer, say 300 iterations. The reason is KNN is too good for 20 shots. Need to run train classifier on 5-shot, OTDD-map network.

- [x] neural + (knn /) a complex classifier: because neural network map is too bad. KNN probably won't work on ImageNet level datasets. So we probably don't use them.

firstly I need to investigate the accuracy of knn. I guess using 100 samples/class for knn is good enough. Use full datasets for small datasets, and half dataset for EMNIST.

Now I see another possible advantage of neural OT over barycentric projection: the generated images are sharper.

- [ ] find how to give the interpolation parameter in the easiest way. Simply use barycentric projection, not neural OT. because the only thing we need is that parameter, which only relies on the mapped data and the distance. But I feel we cannot put it in this conference version.

### 9/28/2022

I find three bugs: 1) didn't change to the same output size, 2) batch size different with training (only for EMNIST), a huge reason. 3) fine_tune_epoch is really number of epoch, not epoch of iterations.

The second third ones can be a huge problem for 20-shot. For 20-shot, need to use 50 batch size and calculate the fine-tune epoch. But luckily these only affect the training process.

For 5-shot today's run, 1) is the only problem. 2) is not because batch size is always 50 > 64, 3) is not because fine-tune epoch = iteration.

Find another bug about blurry barycentric mapping, it's because I pad 1e-4 to all the tensors. This would accumulate a lot in high dimension. But I find this is beneficial for USPS test dataset. If I remove this padding, the result is worse for USPS.

### 9/29/2022

- [x] check the shuffle or not.

- [x] switch to 5-shot interpolation parameter. after workshop. interpolation parameter is nearly all the same 1e-3 difference.

- [x] do mnistm

- [ ] correlation.

- [no] pcam.

- [ ] add discussion about heuristic OTDD and heuristic (2,Q)-distance.

### 10/6/2022

1. fix a unified dataset naming format.
2. give me two more datasets.
3. ask how to add the test dataset, no test dataset.
4. I need a dictionary with the full dataset number of class

I'm at the pretrain_classifier step.

### 10/7/2022

In comparison, training or 1-nn on pcam are done, need to add

- [ ] pretraining on imagenet, and test on pcam?

1. fix the classifier structure.
2. fix the classifier naming format.

I just realize one bug! I should unify all test datasets, now mixup stuff is using test datasets, but knn is using train dataset!!! Damn!!

### 10/8/2022

I wanna figure out the train and test datasets, which I'm using.

All transfer learning methods: train dataset [5, 60000 -5], test dataset. Train on knn of train dataset, test on test dataset.

Train on few-shot directly: train on 5-shot data only, test on test dataset.

knn method: wrong! It's doing knn on train dataset, and test it on test dataset.
