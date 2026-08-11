#include <iostream>

int test()
{
    int res = 0;
    for (int i = 1; i <= 5; i++)
    {
        res += i;
        std::cout << "i = " << i << std::endl;
    }

    return res;

}

int main()
{
    const auto lang = "C++";
    std::cout << "Hello and welcome to " << lang << "!\n";


    std::cout << "i = " << test() << std::endl;

    return 0;}


