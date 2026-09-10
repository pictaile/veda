#ifndef VEDA_GRADIENT_CHECK_H
#define VEDA_GRADIENT_CHECK_H

#include "Variable.h"

#include <functional>
#include <string>

namespace veda::autograd
{

// Central finite differences — the instrument every backward in this epic is checked with.
//
//     dL/dx_i ~= (L(x + h e_i) - L(x - h e_i)) / 2h
//
// Built in the first Build task rather than added at the end, on purpose: a backward that is wrong
// by a constant factor still trains, just worse, and nothing else in the system will ever complain
// (risk R8). Every gradient added from here on is checked the moment it is written.
//
// Two failure modes, both from E3.S1.T1: too small an h and the subtraction is rounding noise, too
// large and the quadratic error term shows. In fp32, h ~ 1e-2 with a relative tolerance of about 2%
// is honest. The comparison is relative per element, because an absolute tolerance is meaningless
// across elements of different magnitude.
//
// It cannot check a gradient that is zero everywhere, and it costs 2n forward passes for n
// elements — so it runs on small tensors in tests, never in training.
struct GradientCheck
{
    bool ok = true;
    double worst_relative = 0.0;
    size_t worst_index = 0;
    double analytic = 0.0;
    double numeric = 0.0;
    std::string message;
};

GradientCheck check_gradient(const std::function<Variable(const Variable&)>& loss_of,
                             const core::Tensor& at, float step = 1e-2f, double tolerance = 2e-2);

} // namespace veda::autograd

#endif //VEDA_GRADIENT_CHECK_H
