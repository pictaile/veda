//
// Created by Константин Охотник on 11.08.2026.
//

#include "Facade.h"

#include <iostream>

#include "BytePairEncoding.h"
#include "WeightsLoader.h"
#include "Tensor.h"
#include "Sampler.h"

using namespace  std;
using namespace  veda::core;

Facade::Facade()
{
    this->bytePairEncoding = BytePairEncoding();
    this->weightsLoader = WeightsLoader();
    this->sampler = Sampler();


}


void Facade::allClassNames()
{
    cout << this->bytePairEncoding.name() << endl;
    cout << this->weightsLoader.name() << endl;
    const Tensor tensor{Shape({2, 3})};
    cout << "Tensor " << tensor.shape() << ", " << tensor.numel() << " elements" << endl;
    cout << this->sampler.name() << endl;
}