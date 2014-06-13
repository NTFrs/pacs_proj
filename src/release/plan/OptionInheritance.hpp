//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

/*
 * OptionBase<dim> ASTRATTA
 */
class OptionBase<dim> {
protected:
	virtual void make_grid();			// da implementare qui
	virtual void steup_system();			// da implementare qui
	virtual void mesh_adapt();			// da implementare qui
	virtual void assemble_system()=0;
	virtual void solve_one_step()=0;
public:
	virtual void run();				// da implementare qui
	virtual void set_refinement(bool);		// da implementare qui
	virtual void set_scale(double);			// da implementare qui
	virtual void set_grid_refinement(unsigned);	// da implementare qui
							// -> al posto che nel costruttore, io metterei questo
	virtual void set_time_step(unsigned);		// da implementare qui
							// -> al posto che nel costruttore, io metterei questo
	// get methods
	// print methods...
	// tutto ciò che è comune va qui.
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

/*
 * OptionBaseLogPrice<dim> ASTRATTA -> assemble_system con coefficienti costanti.
 */
class OptionBaseLogPrice: public OptionBase<dim> {
protected:
	virtual void assemble_system();		// da implementare qui
	virtual void solve_one_step()=0;
};

/*
 * OptionBaseLog<dim> ASTRATTA -> assemble_system con coefficienti non costanti.
 */
class OptionBasePrice: public OptionBase<dim> {
protected:
	virtual void assemble_system();		// da implementare qui
	virtual void solve_one_step()=0;
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

/*
 * EuropeanOptionLogPrice<dim> -> chiama UMFPACK
 */
class EuropeanOptionLogPrice: public OptionBaseLogPrice<dim> {
protected:
	virtual void solve_one_step();		// da implementare qui
};

/*
 * AmericanOptionLogPrice<dim> -> chiama PSOR
 */
class AmericanOptionLogPrice: public OptionBaseLogPrice<dim> {
protected:
	virtual void solve_one_step();		// da implementare qui
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

/*
 * EuropeanOptionPrice<dim> -> chiama UMFPACK
 */
class EuropeanOptionPrice: public OptionBasePrice<dim> {
protected:
	virtual void solve_one_step();		// da implementare qui
};

/*
 * AmericanOptionPrice<dim> -> chiama PSOR
 */
class AmericanOptionPrice: public OptionBasePrice<dim> {
protected:
	virtual void solve_one_step();		// da implementare qui
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

// Al posto di passare al costruttore delle opzioni TUTTI i parametri del mercato
// e dell'opzione, non è meglio creare un oggetto del genere e metterci dentro
// spot, strike, tasso ecc ecc?
class OptionParameter<dim> {
private:
	double K;
	double S1;
	double S2;
	double rho;
	double r;
	
public:
	OptionParameter<1> (K, S, r);
	OptionParameter<2> (K, S1, S2, rho, r);
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

// Factory
class OptionGenerator {
public:
	// Constructor 1d
	OptionBase<1>* CreateOption(CoeffType coeff, ExerciseType type, model * m, OptionParameter * p) {
		if (coeff==LogPrice){
			if (type==US)
				return new AmericanOptionLogPrice<1>(...);
			else
				return new EuropeanOptionLogPrice<1>(...);
		}
		else {
			if (type==US)
				return new AmericanOptionPrice<1>(...);
			else
				return new EuropeanOptionPrice<1>(...);
		}
	}
	
	// Constructor 2d
	OptionBase<2>* CreateOption(CoeffType coeff, ExerciseType type, model * m1, model * m2, OptionParameter * p) {
		if (coeff==LogPrice){
			if (type==US)
				return new AmericanOptionLogPrice<2>(...);
			else
				return new EuropeanOptionLogPrice<2>(...);
		}
		else {
			if (type==US)
				return new AmericanOptionPrice<2>(...);
			else
				return new EuropeanOptionPrice<2>(...);
		}
	}
};

