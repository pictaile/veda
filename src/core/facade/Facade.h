//
// Created by Константин Охотник on 11.08.2026.
//

#ifndef VEDA_FACADE_H
#define VEDA_FACADE_H

#include "BytePairEncoding.h"
#include "WeightsLoader.h"
#include "Tensor.h"
#include "Sampler.h"

class Facade
{
public:
    Facade();
    void allClassNames();

private:
    BytePairEncoding bytePairEncoding;
    WeightsLoader weightsLoader;
    Tensor tensor;
    Sampler sampler;

};


#endif //VEDA_FACADE_H
