#ifndef __option_types_hpp
#define __option_types_hpp

enum class OptionType
{
        Call,
        Put
};

enum class ExerciseType
{
        EU,
        US
};

enum class Transformation {
        Price,
        LogPrice
};

#endif