#include <iostream>
#include "Facade.h"
// #include "facade/Facade.h"

int main()
{
    const auto lang = "C++";

    Facade facade;
    facade.allClassNames();

    std::cout << "end" <<  std::endl;

    return 0;
}


