## Recommended approach for saving a model

There are two main approaches for serializing and restoring a model.

The first (recommended) saves and loads only the model parameters:

```
torch.save(the_model.state_dict(), PATH)
```

Then later:

```
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

The second saves and loads the entire model:

```
torch.save(the_model, PATH)
```

Then later:

```
the_model = torch.load(PATH)
```

However in this case, the serialized data is bound to the specific classes and the exact directory structure used, so it can break in various ways when used in other projects, or after some serious refactors.

– [dontloo](https://stackoverflow.com/users/3041068/dontloo)

According to @smth [discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/…](https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/7) model reloads to train model by default. so need to manually call the_model.eval() after loading, if you are loading it for inference, not resuming training. – [WillZ](https://stackoverflow.com/users/3123992/willz)

Also don't try to save torch.save(model.parameters(), filepath). The model.parameters() is just the generator object.

On the other side, torch.save(model, filepath) saves the model object itself, but keep in mind the model doesn't have the optimizer's state_dict. Check the other excellent answer by @Jadiel de Armas to save the optimizer's state dict.

– [prosti](https://stackoverflow.com/users/5884955/prosti)




If you want to save the model and wants to resume the training later:

#### Single GPU: Save:

```
state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
}
savepath='checkpoint.t7'
torch.save(state,savepath)
```

Load:

```
checkpoint = torch.load('checkpoint.t7')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
```

#### Multiple GPU: Save

```
state = {
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
}
savepath='checkpoint.t7'
torch.save(state,savepath)
```

Load:

```
checkpoint = torch.load('checkpoint.t7')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']

#Don't call DataParallel before loading the model otherwise you will get an error

model = nn.DataParallel(model) #ignore the line if you want to load on Single GPU
```

– [joy-mazumder](https://stackoverflow.com/a/61941234)
