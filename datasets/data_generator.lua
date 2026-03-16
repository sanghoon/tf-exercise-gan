#Uses  space separated specs.txt to output input.txt.
#specs need to have a header row with four columns

require 'torch'
require 'nn'
require 'optim'
require 'pl'

opt={
    folder='',
    data_name='',
    num_samples=768000,
    filename='specs.txt',
    out_name='input.txt',
    t7_filename='data.t7'
}

function tablelength(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local d=data.read(paths.concat(opt.folder,opt.data_name,opt.filename))
local ndim=1
local colnames=d:column_names()
if tablelength(colnames)==4 then
    ndim=2
elseif tablelength(colnames)==3 then
    ndim=1
end
print('ndim '..ndim)

local ncentres=tablelength( d:column_by_name(colnames[1]) )

local centres=torch.Tensor(ncentres,ndim)
local std_dev=torch.Tensor(ncentres)
local densities=torch.Tensor(ncentres)

for i=1,ncentres do
    for j=1,ndim do
        centres[i][j]=tonumber(d:column_by_name(colnames[j])[i])
    end
    std_dev[i]=tonumber(  d:column_by_name(colnames[ndim+1])[i] )
    densities[i]=tonumber(  d:column_by_name(colnames[ndim+2])[i] )
end

paths.mkdir(paths.concat(opt.folder,opt.data_name))
rand_indices=torch.multinomial(densities, opt.num_samples,true )
local data=torch.Tensor(opt.num_samples,ndim)
file=io.open(paths.concat(opt.folder,opt.data_name,opt.out_name) ,'w')
io.output(file)
for i=1,opt.num_samples do
    local k=rand_indices[i]
    local point=torch.Tensor(ndim)
    for j=1,ndim do
        point[j]=torch.normal(0,std_dev[k])+centres[k][j]
    end
    data[i]=point
    if ndim==1 then
        io.write(string.format('%d %f\n',0,point[1]))
    elseif ndim==2 then
        io.write(string.format('%d %f %f\n',0,point[1],point[2]))
    end
end
io.close(file)

torch.save( paths.concat(opt.folder,opt.data_name,opt.t7_filename),data)
