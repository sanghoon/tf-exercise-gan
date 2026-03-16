#creates specs.txt for data_generator.lua
R=5
ncentres=6
distC=100
ncircles=3
std_dev=0.1
density=1

for i=1,ncentres do
    for j=1,ncircles do
        print(R*math.cos((2*i*math.pi)/ncentres)+distC*math.cos((2*j*math.pi)/ncircles) ..' '..R*math.sin((2*i*math.pi)/ncentres)+distC*math.sin((2*j*math.pi)/ncircles))
    end
end

