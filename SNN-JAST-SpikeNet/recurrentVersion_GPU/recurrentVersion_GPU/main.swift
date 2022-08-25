//
//  main.swift
//  recurrentVersion_GPU
//
//  Created by Aditya on 25/08/2022.
//

import MetalKit

typealias DType = Int32

// Network size parameters
let numIn:    DType = 10
let numOut:   DType = 10
let numAct:   DType = 7
let numConnx: DType = 4

// Neuron wipe parameters
let spikeTh: DType = 4
let neurWipe: DType = 1
let wipeTh: DType = 2
let maxSpikes: DType = 10

// Other parameters
let verbose: DType = 2  // no print for verbose zero
let copyRes: Bool  = true  // copy results bool (set true when verbose > 0)




struct OutNeuron {

    var connxArr: [DType]
}


var outNeurArr: [OutNeuron] = []
var inpNeurArr: [[DType]] = []

// Make seed generator (optional for comparing)
struct RandomNumberGeneratorWithSeed: RandomNumberGenerator {
    init(seed: Int) {
        // Set the random seed
        srand48(seed)
    }

    func next() -> UInt64 {
        // drand48() returns a Double, transform to UInt64
        return withUnsafeBytes(of: drand48()) { bytes in
            bytes.load(as: UInt64.self)
        }
    }
}


var generator = RandomNumberGeneratorWithSeed(seed: 900)


// Populate the output Neurons array with random connections
for _ in 0..<numOut {

    outNeurArr.append( OutNeuron( connxArr: (0..<numConnx).map({_ in DType.random(in: -numIn..<numOut, using: &generator)}) ) )

}

if (verbose == 2) {
    print(outNeurArr)
}



// Store the input connections list
var inpConnx: [[DType]] = Array(repeating: [], count: Int(numIn))
var rConnx: [[DType]] = Array(repeating: [], count: Int(numOut))


for i in 0..<numOut {
    for j in 0..<numConnx {

        let ind = outNeurArr[Int(i)].connxArr[Int(j)]
        if (ind < 0) {
            inpConnx[Int(ind + numIn)].append(i)
        }
        else {
            rConnx[Int(ind)].append(i)
        }
    }
}

print("Input's Connections: \(inpConnx)")
print("Recurrent Connections: \(rConnx)")

// Make activation function to create random activation vector
func makeActVect(_ numAct: DType,_ numIn: DType) -> [DType] {

    return (1...numAct).map( {_ in DType.random(in: -numIn..<0, using: &generator)} )

}



// Create zero counter list and size array to be sent to the GPU
var contrArr: [DType] = Array<DType>(repeating: 0, count: Int(numOut / 8) + 1 )

// Start timer
let start = CFAbsoluteTimeGetCurrent()


// Define spike propagation function

func propagate(_ inpNeurArr: [[DType]],_ contrArr: [DType]) -> MTLBuffer? {
    
    
    let device = MTLCreateSystemDefaultDevice()

    let commandQueue = device?.makeCommandQueue()

    let getFunctionLibrary = device?.makeDefaultLibrary()

    let actInp = getFunctionLibrary?.makeFunction(name: "actInp")

    var computePipelineState: MTLComputePipelineState!

    do {
        computePipelineState = try device?.makeComputePipelineState(function: actInp!)
    } catch {
        print(error)
    }


    // Make all memory buffers
    let inpArrBuff = device?.makeBuffer(bytes: Array(inpNeurArr.joined()),
                                        length: MemoryLayout<DType>.stride*Array(inpNeurArr.joined()).count,
                                        options: .storageModeShared)

    let contrArrBuff = device?.makeBuffer(bytes: contrArr,
                                          length: MemoryLayout<DType>.stride*contrArr.count,
                                          options: .storageModeShared)



    // Encode command buffer
    let commandBuff = commandQueue?.makeCommandBuffer()

    let commandEncoder = commandBuff?.makeComputeCommandEncoder()
    commandEncoder?.setComputePipelineState(computePipelineState)

    // Set buffers
    commandEncoder?.setBuffer(inpArrBuff, offset: 0, index: 0)
    commandEncoder?.setBuffer(contrArrBuff, offset: 0, index: 1)


    let w = computePipelineState.threadExecutionWidth
    let thGRW = Array(inpNeurArr.joined()).count
    let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
    let threadsPerGrid = MTLSize(width: thGRW,
                                      height: 1,
                                      depth: 1)


    commandEncoder?.dispatchThreads(threadsPerGrid,
                                               threadsPerThreadgroup: threadsPerThreadgroup)

    commandEncoder?.endEncoding()
    commandBuff?.commit()

    commandBuff?.waitUntilCompleted()
    
    return contrArrBuff
}





// Define process function to apply NeurWipe to the counter and look for Spikes!

func process(_ contrArrBuff: MTLBuffer?,_ spikeList: [DType]) -> (MTLBuffer?, MTLBuffer?) {
    
    let device = MTLCreateSystemDefaultDevice()

    let commandQueue = device?.makeCommandQueue()

    let getFunctionLibrary = device?.makeDefaultLibrary()

    let procssInp = getFunctionLibrary?.makeFunction(name: "procssInp")

    var computePipelineState: MTLComputePipelineState!

    do {
        computePipelineState = try device?.makeComputePipelineState(function: procssInp!)
    } catch {
        print(error)
    }


    // Encode command buffer
    let commandBuff = commandQueue?.makeCommandBuffer()

    let commandEncoder = commandBuff?.makeComputeCommandEncoder()
    commandEncoder?.setComputePipelineState(computePipelineState)


    let neurWipeArr: [DType] = [neurWipe, wipeTh, spikeTh, maxSpikes]

    let neurWipeBuff = device?.makeBuffer(bytes: neurWipeArr,
                                        length: MemoryLayout<DType>.stride*neurWipeArr.count,
                                        options: .storageModeShared)

    let spikeListBuff = device?.makeBuffer(bytes: spikeList,
                                        length: MemoryLayout<DType>.stride*spikeList.count,
                                        options: .storageModeShared)

    // Set buffers

    commandEncoder?.setBuffer(contrArrBuff, offset: 0, index: 0)
    commandEncoder?.setBuffer(neurWipeBuff, offset: 0, index: 1)
    commandEncoder?.setBuffer(spikeListBuff, offset: 0, index: 2)



    let w = computePipelineState.threadExecutionWidth
    let thGRW = Int(numOut)
    let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
    let threadsPerGrid = MTLSize(width: thGRW,
                                      height: 1,
                                      depth: 1)


    commandEncoder?.dispatchThreads(threadsPerGrid,
                                               threadsPerThreadgroup: threadsPerThreadgroup)

    commandEncoder?.endEncoding()
    commandBuff?.commit()

    commandBuff?.waitUntilCompleted()
    
    
    return (contrArrBuff, spikeListBuff)
    

}


// Randomly activate inputs
var actVect: [DType] = makeActVect(numAct, numIn)

func makeInpNeurArr(_ actVect: [DType],_ inpConnx: [[DType]],_ rConnx: [[DType]]) -> [[DType]] {
    
    var inpNeurArr: [[DType]] = []
    
    for ind in actVect {
        
        if (ind < 0) {
            inpNeurArr.append(inpConnx[Int(ind + numIn)])
        }
        else {
            inpNeurArr.append(rConnx[Int(ind)])

        }
        

    }
    
    return inpNeurArr
    
}







let num_steps = 5

for _ in 0..<num_steps {
    
    
    inpNeurArr = makeInpNeurArr(actVect, inpConnx, rConnx)
    
    if verbose != 0 {

    print("Activated input indices: \(actVect)")
    print("Connection array of activated inputs: \(inpNeurArr)")

    }
    
    
    // Run the first kernel to register spike counts from the activated inputs //
    var contrArrBuff: MTLBuffer? = propagate(inpNeurArr, contrArr)


    if copyRes == true {
        var contrArrPointer = contrArrBuff?.contents().bindMemory(to: DType.self, capacity: MemoryLayout<DType>.stride*contrArr.count)

        // Print out the output neurons after spiking
        for ind in 0..<contrArr.count {

            contrArr[ind] = contrArrPointer!.pointee
            contrArrPointer = contrArrPointer?.advanced(by: 1)
        }
    }

    if verbose != 0 {
        print("Counter Array: \(contrArr)")
    }
    
    
    // Run the second kernel to process spikes and generate spike list  //
    var spikeList: [DType] = Array<DType>(repeating: 0, count:  Int(maxSpikes)+1)
    var spikeListBuff: MTLBuffer?
    
    (contrArrBuff, spikeListBuff) = process(contrArrBuff, spikeList)
    
    var spikeListPointer = spikeListBuff?.contents().bindMemory(to: DType.self, capacity: MemoryLayout<DType>.stride*spikeList.count)
    
    let lenSpike: DType = spikeListPointer!.pointee
    
    for i in 1...lenSpike {
        
        let indPointer = spikeListPointer?.advanced(by: Int(i))
        let ind: DType = indPointer!.pointee
        actVect.append(ind)
        
    }

    if copyRes == true {
        
        var contrArrPointer = contrArrBuff?.contents().bindMemory(to: DType.self, capacity: MemoryLayout<DType>.stride*contrArr.count)

        // Print out the counter array
        for ind in 0..<contrArr.count {

            contrArr[ind] = contrArrPointer!.pointee
            contrArrPointer = contrArrPointer?.advanced(by: 1)

        }
        
        
        // Store spike list in an array
        for ind in 0..<spikeList.count {

            spikeList[ind] = spikeListPointer!.pointee
            spikeListPointer = spikeListPointer?.advanced(by: 1)
        }
        
        
    }

    if verbose != 0 {
        print("Counter Array after NeurWipe: \(contrArr)")
        print("SpikeList generated: \(spikeList)")
    }
    
    print("\n")
    
    
}


// End timer
let end = CFAbsoluteTimeGetCurrent()
// Print out the time elapsed
print("Time elapsed in GPU compute = \(end - start)")
