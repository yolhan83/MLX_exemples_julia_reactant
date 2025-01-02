using MLDatasets,Lux,Reactant,Enzyme,Random,OneHotArrays,MLUtils,Optimisers,Statistics
Reactant.set_default_backend("gpu")

const rng = Random.MersenneTwister(123)
const dev = xla_device()


# Load the MNIST dataset
mnist_train = MLDatasets.MNIST(:train)
mnist_test = MLDatasets.MNIST(:test)

# Make A MLP model
function make_mlp(num_layer::Int,input_dim ::Int,hidden_dim::Int,output_dim::Int)
    model = Chain(
        Dense(input_dim,hidden_dim,relu),
        [Dense(hidden_dim,hidden_dim,relu) for i in 1:num_layer-2]...,
        Dense(hidden_dim,output_dim)
    )
    ps,st = Lux.setup(rng,model) |> dev
    return model,ps,st
end

function loss_fn(model, ps, st, (x, y))
    ŷ, stₙ = model(x, ps, st)
    return CrossEntropyLoss(;agg=mean,logits=true)(ŷ, y), stₙ, (;)
end

function main()
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 128
    num_epochs = 32
    learning_rate = 1e-3
    imgs_train = reshape(mnist_train.features,
        (
            size(mnist_train.features,1)*size(mnist_train.features,2),
            size(mnist_train.features,3)
        )
        )
    imgs_test = reshape(mnist_test.features,
        (
            size(mnist_test.features,1)*size(mnist_test.features,2),
            size(mnist_test.features,3)
        )
        )
    class = 0:9
    targets_train = onehotbatch(mnist_train.targets,class)

    model,ps,st = make_mlp(num_layers,28*28,hidden_dim,num_classes)

    dl_train = DataLoader((imgs_train,targets_train),batchsize=batch_size,shuffle=true,partial=false)|> dev
    opt = Training.TrainState(model,ps,st,Adam(learning_rate))
    for epoch in 1:num_epochs
        for (i,(x,y)) in enumerate(dl_train)
            _,loss,_,opt = Training.single_train_step!(
                AutoEnzyme(),
                loss_fn,
                (x,y),
                opt
            )
            if i==1
                @info "Epoch: $epoch, Loss: $loss"
            end
        end
    end
    ps = opt.parameters
    targets = mnist_test.targets
    modelcomp = @compile model(dev(imgs_test),ps,st)
    ŷ,_ = modelcomp(dev(imgs_test),ps,st)
    ŷ = argmax.(eachcol(Array(ŷ))) .- 1
    acc = mean(ŷ .== targets)*100
    @info "Accuracy: $acc"
    return opt
end

opt = main()