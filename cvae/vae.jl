struct UpsamplingConv{C1} <: Lux.AbstractLuxWrapperLayer{:conv}
    conv :: C1
end
function UpsamplingConv(in_channels,out_channels,kernel_size,stride,pad)
    conv = Lux.Conv((kernel_size,kernel_size),in_channels=>out_channels,pad = pad,stride = stride)
    return UpsamplingConv(conv)
end
function (UC ::UpsamplingConv)(x,ps,st)
    res,_ = UC.conv(upsample_nearest(x,(2,2)),ps,st)
    return res,st
end

struct Encoder{C1,C2,C3,BN1,BN2,BN3,PMu,PLV} <: Lux.AbstractLuxContainerLayer{(:conv1,:conv2,:conv3,:bn1,:bn2,:bn3,:proj_mu,:proj_log_var)}
    conv1 ::C1
    conv2::C2
    conv3::C3
    bn1::BN1
    bn2::BN2
    bn3::BN3
    proj_mu::PMu
    proj_log_var::PLV
end

function Encoder(num_latent_dims,image_shape,max_num_filters)
    num_filters_1 = max_num_filters ÷ 4
    num_filters_2 = max_num_filters ÷ 2
    num_filters_3 = max_num_filters
    channel = image_shape[end]
    conv1 = Lux.Conv((3,3),channel=>num_filters_1,stride=2,pad = 1)
    conv2 = Lux.Conv((3,3),num_filters_1=>num_filters_2,stride=2,pad = 1)
    conv3 = Lux.Conv((3,3),num_filters_2=>num_filters_3,stride=1,pad = 1)
    bn1 = Lux.BatchNorm(num_filters_1)
    bn2 = Lux.BatchNorm(num_filters_2)
    bn3 = Lux.BatchNorm(num_filters_3)
    out_shape = prod(dim ÷ 4 for dim in image_shape[1:end-1]) * num_filters_3
    proj_mu = Lux.Dense(out_shape,num_latent_dims)
    proj_log_var = Lux.Dense(out_shape,num_latent_dims)
    return Encoder(conv1,conv2,conv3,bn1,bn2,bn3,proj_mu,proj_log_var)
end
function (E::Encoder)(z,ps,st)
    x,_ = E.conv1(z,ps.conv1,st.conv1)
    x,_ = E.bn1(x,ps.bn1,st.bn1)
    x = relu.(x)
    # @info size(x),E.conv2
    x,_ = E.conv2(x,ps.conv2,st.conv2)
    x,_ = E.bn2(x,ps.bn2,st.bn2)
    x = relu.(x)
    x,_ = E.conv3(x,ps.conv3,st.conv3)
    x,_ = E.bn3(x,ps.bn3,st.bn3)
    x = relu.(x)
    # @info size(x)
    x = reshape(x,(size(x,1)*size(x,2)*size(x,3),size(x,4))) 
    mu,_ = E.proj_mu(x,ps.proj_mu,st.proj_mu)
    log_var,_ = E.proj_log_var(x,ps.proj_log_var,st.proj_log_var)
    sigma = exp.(0.5f0 .* log_var)
    eps = randn(Float32,size(sigma))
    x = mu .+ sigma .* eps
    return x,mu,log_var,st
end

struct Decoder{L1,UC1,UC2,UC3,BN1,BN2} <: Lux.AbstractLuxContainerLayer{(:lin1,:upconv1,:upconv2,:upconv3,:bn1,:bn2)}
    lin1 ::L1
    upconv1 ::UC1
    upconv2  ::UC2
    upconv3  ::UC3
    bn1 ::BN1
    bn2  ::BN2
    input_shape ::Tuple{Int,Int,Int}
    max_num_filters ::Int
end
function Decoder(num_latent_dims,image_shape,max_num_filters)
    num_filters_1 = max_num_filters
    num_filters_2 = max_num_filters ÷ 2
    num_filters_3 = max_num_filters ÷ 4
    input_shape = vcat([dim÷4 for dim in image_shape[1:end-1]],num_filters_1)
    flshap = prod(input_shape)
    lin1 = Lux.Dense(num_latent_dims,flshap)
    upconv1 = UpsamplingConv(num_filters_1,num_filters_2,3,1,1)
    upconv2 = UpsamplingConv(num_filters_2,num_filters_3,3,1,1)
    channel = image_shape[end]
    upconv3 = UpsamplingConv(num_filters_3,channel,3,2,1)
    bn1 = Lux.BatchNorm(num_filters_2)
    bn2 = Lux.BatchNorm(num_filters_3)
    return Decoder(lin1,upconv1,upconv2,upconv3,bn1,bn2,Tuple(input_shape), max_num_filters)
end
function (D::Decoder)(z,ps,st)
    x,_ = D.lin1(z,ps.lin1,st.lin1)
    s = size(x)
    x = reshape(x,(D.input_shape[1],D.input_shape[2],D.max_num_filters,s[end]))
    x = leakyrelu.(D.bn1(D.upconv1(x,ps.upconv1,st.upconv1)[1],ps.bn1,st.bn1)[1])
    x = leakyrelu.(D.bn2(D.upconv2(x,ps.upconv2,st.upconv2)[1],ps.bn2,st.bn2)[1])
    x = sigmoid.(D.upconv3(x,ps.upconv3,st.upconv3)[1])
    return x,st
end

struct VAE{E<:Lux.AbstractLuxContainerLayer,D} <: Lux.AbstractLuxContainerLayer{(:encoder,:decoder)}
    encoder ::E
    decoder ::D
    num_latent_dims ::Int
end
function VAE(num_latent_dims::Int,image_shape,max_num_filters::Int)
    encoder = Encoder(num_latent_dims,image_shape,max_num_filters)
    decoder = Decoder(num_latent_dims,image_shape,max_num_filters)
    return VAE(encoder,decoder,num_latent_dims)
end
function (V::VAE)(x,ps,st)
    z,mu,log_var,_ = V.encoder(x,ps.encoder,st.encoder)
    x̂,_ = V.decoder(z,ps.decoder,st.decoder)
    return x̂,mu,log_var,st
end
