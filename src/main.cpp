#include <iostream>
#include "Facade.h"
// #include "facade/Facade.h"

int main()
{
    const auto lang = "C++";

    Facade facade;
    facade.examples();

    facade.example_create_tensor();
    std::cout << "end" <<  std::endl;

    return 0;
}


