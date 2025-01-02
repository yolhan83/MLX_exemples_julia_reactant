using Lux,Reactant,Enzyme,Random,MLDatasets,MLUtils,Optimisers,Statistics,Tullio,LoopVectorization,Plots,Images,Zygote

include("cvae/vae.jl")

Reactant.set_default_backend("gpu")
const rng = Random.MersenneTwister(123)
const dev = xla_device() # xla_device() when it will work or if you want to see the error


function grid_image_from_batch(image_batch, num_rows)
    H, W, C, B = size(image_batch)
    num_cols = div(B, num_rows)
    grid_image = zeros(H * num_rows, W * num_cols, C)

    for b in 1:B
        row = div(b - 1, num_cols) + 1 
        col = (b - 1) % num_cols + 1  
        a1 = (row - 1) * H + 1 
        b1 = row * H
        a2 = (col - 1) * W + 1
        b2 = col * W 
        for i in a1:b1, j in a2:b2, k in 1:C
            grid_image[i, j, k] = image_batch[i - a1 + 1, j - a2 + 1, k, b]
        end
    end
    return grid_image
end


function loss_fn(model,ps,st,(X,))
    X_recon,mu,logvar,_ = model(X,ps,st)
    recon_loss = MSELoss()(X_recon,X)
    kl_div = -0.5f0*sum(1.0f0 .+ logvar .- mu.*mu .- exp.(logvar))
    return recon_loss + kl_div,st,(;)
end

function reconstruct(model,ps,st,image_batch)
    recon = model(image_batch,ps,st)[1]
    paired_images = cat(image_batch, recon,dims=4)
    return grid_image_from_batch(paired_images, 100)
end

function generate(model,ps,st,num_samples=128)
    z = randn(rng,(model.num_latent_dims,num_samples)) |> dev
    m = @compile model.decoder(z,ps.decoder,st.decoder)
    nim = m(z,ps.decoder,st.decoder)[1]
    return grid_image_from_batch(nim, 8)
end

function main()
    mnist_train = MLDatasets.MNIST(:train)
    mnist_test = MLDatasets.MNIST(:test)
    batch_size = 128
    num_epochs = 32
    learning_rate = 1f-3
    imshape = (28,28,1) 
    imgs_train = reshape(mnist_train.features,
        (
            size(mnist_train.features,1),
            size(mnist_train.features,2),
            1,
            size(mnist_train.features,3)
        )
        ) |> f32
    imgs_test = reshape(mnist_test.features,
        (
            size(mnist_test.features,1),
            size(mnist_test.features,2),
            1,
            size(mnist_test.features,3)
        )
        )[:,:,:,1:10] |> f32
    model = VAE(64,imshape,128)
    ps,st = Lux.setup(rng,model) |> dev
    dl_train = DataLoader((imgs_train,),batchsize=batch_size,shuffle=true,partial=false)|> dev
    opt = Training.TrainState(model,ps,st,Adam(learning_rate))
    for epoch in 1:num_epochs
        for (i,(x,)) in enumerate(dl_train)
            _,loss,_,opt = Training.single_train_step!(
                AutoEnzyme(),
                loss_fn,
                (x,),
                opt
            )
            if i==1
                @info "Epoch: $epoch, Loss: $loss"
            end
        end
    end
    ps = opt.parameters
    m = @compile model(dev(imgs_test),ps,st)
    recon = m(dev(imgs_test),ps,st)[1]#model(dev(imgs_test),ps,st)[1]#
    paired_images = cat(imgs_test, collect(recon),dims=1)
    grid_image = grid_image_from_batch(paired_images, 5)
    fig = plot(colorview(Gray, grid_image[:,:,1]')) 
    return fig,model,ps,st
end

fig,model,ps,st = main();
fig # obvious bad reconstruction (less bad than expected) on cpu with only one batch
