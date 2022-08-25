//
//  main.swift
//  outputConnectionFileGen
//
//  Created by Aditya on 26/08/2022.
//

import Foundation
import MetalKit

typealias DType = Int32

let numIn:    DType = 10000000
let numOut:   DType = 100000000
let numConnx: DType = 4


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



var generator = RandomNumberGeneratorWithSeed(seed: 7)


// User should modify this folder path to where benchmark folder is saved in local device
let dirPath = "/Users/aditya/Desktop/SNN-JAST-SpikeNet"

// Generate output connections file
let fileName = dirPath + "/benchmark/Connections_8.bin"
var fpOut: UnsafeMutablePointer<FILE>?

fpOut = fopen(fileName, "wb")

for _ in 0..<numOut {

    let tmp = (0..<numConnx).map({_ in DType.random(in: 0..<numIn, using: &generator)})
    let buffer = UnsafeMutablePointer<DType>.allocate(capacity: tmp.count)
    buffer.initialize(from: tmp, count: tmp.count)
    let size = tmp.count * MemoryLayout<DType>.stride
    fwrite(buffer, size, 1, fpOut)
//    print(tmp)

    buffer.deinitialize(count: size)
    free(buffer)

}

fclose(fpOut)
