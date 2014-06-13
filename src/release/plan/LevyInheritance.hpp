//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

// le classi con coeff costanti chiamano questa
class LevyIntegralLogPrice<dim>{
protected:
	double alpha1;
	double alpha2;
	
	dealii::Vector<double> J1;
	dealii::Vector<double> J2;
public:
	void evaluate_part1();		// calcola alpha1 in 1d, alpha1, alpha2 in 2d
	void evaluate_part2();		// calcola J1 in 1d, J1, J2 in 2d
	
	void get_part1(double &alpha_);						// in 1d chiamo questo
	void get_part1(double &alpha1_, double &alpha_2);			// in 2d chiamo questo
	
	void get_J(dealii::Vector<double> J1);					// in 1d chiamo questo
	void get_J(dealii::Vector<double> J1, dealii::Vector<double> J2);	// in 2d chiamo questo
};

// le classi con coeff costanti chiamano questa
class LevyIntegralPrice<dim>{
protected:
	double alpha1;
	double alpha2;
	
	dealii::Vector<double> J1;
	dealii::Vector<double> J2;
public:
	void evaluate_part1();		// calcola alpha1 in 1d, alpha1, alpha2 in 2d
	void evaluate_part2();		// calcola J1 in 1d, J1, J2 in 2d
	
	void get_part1(double &alpha_);						// in 1d chiamo questo
	void get_part1(double &alpha1_, double &alpha_2);			// in 2d chiamo questo
	
	void get_J(dealii::Vector<double> J1);					// in 1d chiamo questo
	void get_J(dealii::Vector<double> J1, dealii::Vector<double> J2);	// in 2d chiamo questo
};

// da qui facciamo ereditare per Merton e Kou, calcolando con gli opportuni nodi gli alpha sia per logS, sia per S, e per J usiamo i nodi di Hermite e Laguerre per LogS, e la quadratura di dealii per S.
