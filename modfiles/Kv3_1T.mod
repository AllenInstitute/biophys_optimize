: Comment: Kv3-like potassium current

NEURON	{
	SUFFIX Kv3_1T
	USEION k READ ek WRITE ik
	RANGE gbar, g, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
	vshift = 0 (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau
	celsius (degC)
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar*m
	ik = g*(v-ek)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
		LOCAL qt
		qt = 2.3 ^ ((celsius - 23) / 10)
	UNITSOFF
		mInf =  1 / (1 + exp(((v - (18.700 + vshift)) / -9.700)))
		mTau =  (0.2 / qt) * 20.0 / (1 + exp(((v - (-46.560 + vshift)) / -44.140)))
	UNITSON
}
