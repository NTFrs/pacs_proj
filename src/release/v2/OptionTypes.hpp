#ifndef __option_types_hpp
#define __option_types_hpp

//! EnumClass for hanling Option Types.
/*! This EnumClass defines the Option Types, Put or Call.
 */
enum class OptionType
{
        Call,
        Put
};

//! EnumClass for hanling Exercise Types.
/*! This EnumClass defines the Exercise Types, European or American
 */
enum class ExerciseType
{
        EU,
        US
};

//! EnumClass for defining the transformation type.
/*! This EnumClass defines the transformation type, Price or LogPrice.
 */
enum class Transformation {
        Price,
        LogPrice
};

#endif