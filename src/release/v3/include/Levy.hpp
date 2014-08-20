#ifndef __levy_hpp
#define __levy_hpp

#include "LevyIntegralBase.hpp"
#include "LevyIntegralLogPrice.hpp"
#include "LevyIntegralPrice.hpp"
#include "LevyIntegralPriceKou.hpp"
#include "LevyIntegralPriceMerton.hpp"
#include "LevyIntegralLogPriceKou.hpp"
#include "LevyIntegralLogPriceMerton.hpp"

#include "EuropeanOptionLogPrice.hpp"
#include "AmericanOptionLogPrice.hpp"
#include "EuropeanOptionPrice.hpp"
#include "AmericanOptionPrice.hpp"

#include "Factory.hpp"

/*! \mainpage A Fem solver for pricing financial derivatives
 * 
 * The present project is a small library that can be used to price financial derivatives on Levy Assets in both one and two dimensions. It provides the functions to price the basic financial objects,  as well as the base structure to easily create new financial objects with few lines of code.
 * 
 * \section install_sec Installing
 * 
 * In order to use this library,  it is only needed to have the deal.ii library and CMake installed. Since it's mostly templates, it is practically a collection of header files, and thus it does not require an installation. It all comes down to using the provided CMakeLists and setting it as necessary. See the section \ref using_sec on how to edit the CMakeLists for basic needs.
 * 
 * \section using_sec Using the library
 * 
 * In order to use this tool, all that is needed is to create a CMakeLists file that tells the CMake utility which file to compile. Basic CMakeLists files are provided in the examples (test 1 to 5). For a new project, it is only needed to set two or three things in this file:
 * \arg Set the target of the project (which would be the main) with \code SET(TARGET "target_name")\endcode where target_name is the name of the main withouth the extension.
 * \arg Set where this library can be found through include_directories(<em>path to the directory with headers</em>)
 * \arg It may be needed to set the additional sources. For example,  this tool uses a source file called `QuadratureRule.cpp` that is specified by setting the variable TARGET_SRC
 * \code# Declare all source files the target consists of:
	SET(TARGET_SRC
	${TARGET}.cpp
	# You can specify additional files here!
	../src/QuadratureRule.cpp
	) \endcode
 *Note that it alredy sets the main file,  so it is not needed to set it explicitly
 *
 * In the default configuration, the main file is in a directory,  while the include and source directories with the library files are just outside it. In that case,  it would only be needed to add the name of the main file and eventual sources.
 * 
 * \subsection basic_use Basic use of the library
 * 
 * An object that represent the most common financial derivative,  a \a Call or \a Put,  is already implemented through the classes EuropeanOptionLogPrice\<dim\> and EuropeanOptionPrice\<dim\>. The difference between them is the transformation used on the equation to solve the problem. Also,  AmericanOptionLogPrice\<dim\> and AmericanOptionPrice\<dim\> are present to price <em>American Put Options</em>. The usage is pretty simple,  just declare a Model,  and declare an option passing the Model (or models) with the Model::get_pointer() method,  as well as the other parameters. Then, run() and get_price().
 * 
 * \subsection create_more_options Advanced use: Creating new financial options
 * 
 * It is possible to create new options for specific financial derivatives. The standard way to do it is to create a class that inherits from OptionBasePrice\<dim\> or OptionBaseLogPrice\<dim\> and implement the solve method in which it is possible to specify different boundary conditions or solvers. Most of the times,  it is only needed to specify different Boundary Conditions, and specify how to use the Integral part computed by the LevyIntegral classes (see LevyIntegralBase\<dim\> ).
 * 
 * Also,  it is possible to add more models besides the three that are already there. In that case,  all that is needed is to inherit from Model class and specify the needed parameters and the density. In that case it may be needed to redefine the method OptionBase\<dim\>::setup_integral() in order to use the specific levy class needed.
 * 
 */

#endif