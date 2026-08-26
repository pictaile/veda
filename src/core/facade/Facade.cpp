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

Facade::Facade()
{
    this->bytePairEncoding = BytePairEncoding();
    this->weightsLoader = WeightsLoader();
    this->tensor = Tensor();
    this->sampler = Sampler();


}


void Facade::allClassNames()
{
    cout << this->bytePairEncoding.name() << endl;
    cout << this->weightsLoader.name() << endl;
    cout << this->tensor.name() << endl;
    cout << this->sampler.name() << endl;
}