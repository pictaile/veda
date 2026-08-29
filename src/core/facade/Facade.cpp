//
// Created by Константин Охотник on 11.08.2026.
//

#include "Facade.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "BFloat16.h"
#include "BytePairEncoding.h"
#include "DType.h"
#include "Sampler.h"
#include "Shape.h"
#include "Storage.h"
#include "Strides.h"
#include "Tensor.h"
#include "WeightsLoader.h"

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

namespace
{

// Width in UTF-8 code points rather than bytes — Cyrillic text would otherwise stretch every
// heading rule and break every column.
size_t display_width(const string& text)
{
    size_t width = 0;
    for (const char byte : text)
    {
        if ((static_cast<unsigned char>(byte) & 0xC0) != 0x80)
        {
            ++width;
        }
    }
    return width;
}

string pad(const string& text, size_t width)
{
    const size_t used = display_width(text);
    return text + string(used < width ? width - used : 1, ' ');
}

void heading(const string& title)
{
    cout << "\n" << title << "\n" << string(display_width(title), '-') << endl;
}

// shape, strides, offset and contiguity — the four things a tensor's description is made of
void describe(const string& label, const Tensor& tensor)
{
    cout << "  " << pad(label, 18)
         << "форма " << pad(tensor.shape().to_string(), 14)
         << "страйди " << pad(to_string(tensor.strides()), 16)
         << "offset " << pad(std::to_string(tensor.offset()), 4)
         << (tensor.is_contiguous() ? "щільний" : "нещільний") << endl;
}

// Prints the elements the way the tensor reads them, not the way they lie in memory.
void print_values(const string& label, const Tensor& tensor)
{
    cout << "  " << pad(label, 18);

    if (tensor.rank() == 0)
    {
        cout << tensor.at({}) << endl;
        return;
    }
    if (tensor.rank() == 1)
    {
        cout << "[ ";
        for (size_t i = 0; i < tensor.shape()[0]; ++i)
        {
            cout << tensor.at({i}) << " ";
        }
        cout << "]" << endl;
        return;
    }

    for (size_t row = 0; row < tensor.shape()[0]; ++row)
    {
        if (row > 0)
        {
            cout << "  " << string(18, ' ');
        }
        cout << "[ ";
        for (size_t col = 0; col < tensor.shape()[1]; ++col)
        {
            cout << tensor.at({row, col}) << " ";
        }
        cout << "]" << endl;
    }
}

// The flat buffer underneath, in memory order — so that "no byte moved" can be seen rather than
// taken on trust.
void print_buffer(const string& label, const Tensor& tensor)
{
    cout << "  " << pad(label, 18) << "[ ";
    for (size_t i = 0; i < tensor.storage()->size(); ++i)
    {
        cout << tensor.storage()->data()[i] << " ";
    }
    cout << "]" << endl;
}

} // namespace

void Facade::example_create_tensor()
{
    Tensor tensor(Shape({9, 9, 3}));
    describe("новий тензор:", tensor);
    cout << "Rank " << tensor.rank() << endl;
    cout << "numel " << tensor.numel() << endl;
}

void Facade::examples()
{
    cout << "\n=== veda::core — що вміє тензор (епік E1) ===" << endl;

    // 1. Creating a tensor: a shape, and storage allocated for it, zero-filled.
    heading("1. Створення тензора");
    Tensor tensor{Shape({2, 3})};
    describe("новий тензор:", tensor);
    print_values("значення:", tensor);
    cout << "  ранг " << tensor.rank() << ", елементів " << tensor.numel()
         << ", сховище на " << tensor.storage()->size() << " float, занулене" << endl;

    // Shape knows nothing about what the dimensions mean — [B, T, D] is a convention of the
    // code that builds it.
    const Shape activation({1, 8, 64});
    cout << "  справжня форма [B=1, T=8, D=64] " << activation
         << " містить " << activation.size() << " float" << endl;

    // 2. Filling it: data() is the only place where element counting becomes pointer arithmetic.
    heading("2. Заповнення через data()");
    for (size_t i = 0; i < tensor.numel(); ++i)
    {
        tensor.data()[i] = static_cast<float>(10 * (i + 1));
    }
    print_values("значення:", tensor);
    print_buffer("буфер:", tensor);

    // 3. Reading by coordinates: offset = sum of index * stride, plus the tensor's own offset.
    heading("3. Читання за координатами");
    cout << "  страйди " << to_string(tensor.strides())
         << " означають: рядок коштує 3 елементи, стовпець — 1" << endl;
    cout << "  at({1, 2}) = 1*3 + 2*1 = зсув 5 -> " << tensor.at({1, 2}) << endl;
    cout << "  at({0, 1}) = 0*3 + 1*1 = зсув 1 -> " << tensor.at({0, 1}) << endl;

    // 4. Views: several tensors describing one buffer. A view is not a snapshot.
    heading("4. В'ю над спільним сховищем");
    auto storage = tensor.storage();
    Tensor whole = Tensor::view(storage, 0, Shape({2, 3}));
    Tensor second_row = Tensor::view(storage, 3, Shape({3}));
    describe("увесь буфер:", whole);
    describe("другий рядок:", second_row);
    print_values("значення:", second_row);
    cout << "  сховище тримають " << storage.use_count()
         << " власники: початковий тензор, два в'ю і ця локальна змінна" << endl;

    second_row.data()[0] = 99.0f;
    cout << "  записуємо 99 через другий рядок..." << endl;
    print_values("увесь буфер:", whole);
    cout << "  запис видно крізь кожне в'ю — копії, яка могла б застаріти, не існує" << endl;
    second_row.data()[0] = 40.0f;

    // 5. reshape: same elements, a different reading rule. Requires contiguity.
    heading("5. reshape — ті самі байти, інша форма");
    const Tensor flat = tensor.reshape(Shape({6}));
    const Tensor three_by_two = tensor.reshape(Shape({3, 2}));
    describe("було (2,3):", tensor);
    describe("-> (6):", flat);
    print_values("значення:", flat);
    describe("-> (3,2):", three_by_two);
    print_values("значення:", three_by_two);
    cout << "  той самий вказівник: " << boolalpha << (flat.data() == tensor.data())
         << " — скопійовано нуль байтів" << endl;

    // The attention head split is exactly this operation.
    const Tensor projection{Shape({1, 8, 1024})};
    const Tensor heads = projection.reshape(Shape({1, 8, 16, 64}));
    cout << "  розділення на голови: " << projection.shape() << " -> " << heads.shape()
         << ", страйди " << to_string(heads.strides()) << endl;

    // 6. transpose: swap two axes by swapping their strides.
    heading("6. transpose — обмін осей");
    const Tensor transposed = tensor.transpose(0, 1);
    describe("до:", tensor);
    describe("після:", transposed);
    print_values("значення:", transposed);
    print_buffer("буфер:", transposed);
    cout << "  буфер недоторканий; змінилось лише правило читання" << endl;
    cout << "  транспонування двічі повертає початкові страйди: "
         << (transposed.transpose(0, 1).strides() == tensor.strides()) << endl;

    // A transposed view is the classic non-contiguous tensor, and reshape must refuse it.
    try
    {
        (void)transposed.reshape(Shape({6}));
    }
    catch (const std::exception& error)
    {
        cout << "  reshape транспонованого в'ю відхилено:" << endl;
        cout << "    " << error.what() << endl;
    }

    // 7. slice: narrow one dimension, shift the offset, keep the strides.
    heading("7. slice — звуження одного виміру");
    const Tensor row = tensor.slice(0, 1, 1);
    const Tensor columns = tensor.slice(1, 1, 2);
    describe("рядок 1:", row);
    print_values("значення:", row);
    describe("стовпці 1..2:", columns);
    print_values("значення:", columns);
    cout << "  зріз по виміру 0 зберігає щільність; зріз по внутрішньому лишає діру "
         << "(пропущене 40)" << endl;

    // Both real uses: the last position's logits, and the growing KV cache prefix.
    const Tensor logits{Shape({1, 8, 32})};
    const Tensor last = logits.slice(1, 7, 1);
    cout << "  останні логіти: " << logits.shape() << " -> " << last.shape()
         << " зі зсувом " << last.offset()
         << (last.is_contiguous() ? ", щільні -> reshape до [B, V] безпечний" : "") << endl;

    const Tensor cache{Shape({1, 2, 16, 4})};
    cout << "  префікс KV-кешу на кроці 3: " << cache.shape() << " -> "
         << cache.slice(2, 0, 3).shape() << " — без переалокації" << endl;

    // 8. Everything composes, because every operation returns a description.
    heading("8. Композиція в'ю");
    const Tensor composed = tensor.transpose(0, 1).slice(0, 1, 2);
    describe("transpose+slice:", composed);
    print_values("значення:", composed);

    // 9. The I/O boundary: dtype names, byte extents, and the bf16 widening.
    heading("9. Завантаження ваг: dtype і bf16 -> fp32");
    const DType dtype = dtype_from_name("BF16");
    cout << "  dtype_from_name(\"BF16\") -> " << dtype_name(dtype)
         << ", " << dtype_size(dtype) << " байти на елемент" << endl;
    cout << "  тензор [1024] у BF16 займає " << byte_length(Shape({1024}), dtype) << " байт;"
         << " таблиця вкладень [151669, 1024] — "
         << byte_length(Shape({151669, 1024}), dtype) << " байт" << endl;

    try
    {
        (void)dtype_from_name("F16");
    }
    catch (const std::exception& error)
    {
        cout << "  непідтримувані типи падають гучно: " << error.what() << endl;
    }

    const vector<uint16_t> raw = {0x3F80, 0xC000, 0x0000, 0x4049, 0xBF80, 0x4000};
    Tensor weights{Shape({2, 3})};
    widen_bf16(raw.data(), weights.data(), weights.numel());
    cout << "  bf16 0x3F80 << 16 = 0x3F800000 -> " << bf16_to_f32(0x3F80)
         << ", 0x4049 -> " << bf16_to_f32(0x4049) << " (π, скільки його вміщує bf16)" << endl;
    print_values("розширені:", weights);
    print_values("транспоновані:", weights.transpose(0, 1));

    cout << "\n  Тензор — це опис пам'яті, а не сама пам'ять." << endl;
}
