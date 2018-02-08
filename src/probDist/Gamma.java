/**
 * Copyright 2018 Felipe Hernández
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software distributed under the 
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

package probDist;

/**
 * @author Felipe Hernández
 */
public class Gamma 
{
	
	// --------------------------------------------------------------------------------------------
	// Static methods
	// --------------------------------------------------------------------------------------------

	/**
	 * Computes the value of the cumulative distribution function of a Gamma probability 
	 * distribution
	 * @param k The shape parameter of the distribution. Must be greater than zero.
	 * @param theta The scale parameter of the distribution. Must be greater than zero.
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double k, double theta, double x)
	{
		return incompleteGammaFunctionP(k, x/theta);
	}
	
	/**
	 * Computes the value of a random variable with a Gamma distribution that has the cumulative 
	 * distribution function value that enters as a parameter. That is, the inverse cumulative 
	 * distribution function value.
	 * @param k The shape parameter of the distribution. Must be greater than zero.
	 * @param theta The scale parameter of the distribution. Must be greater than zero.
	 * @param p The value of the cumulative distribution function
	 * @return The inverse cumulative distribution function value
	 */
	public static double computeInvCDF(double k, double theta, double p)
	{		
		return 0.5 * theta * pointChi2(2*k, p);
	}
	
	/**
	 * Computes the value of the Gamma function (i.e. the extension of the factorial function to 
	 * real numbers)
	 * @param alpha The parameter of the function. Must be greater than zero.
	 * @return The value of the Gamma function
	 */
	public static double computeGamma(double alpha)
	{
		return Math.exp(computeLnGamma(alpha));
	}

	/**
	 * Computes the log Gamma function: <i>ln(gamma(alpha))</i> for alpha > 0. Accurate to 10 
	 * decimal places.
	 * @param alpha The parameter of the function
	 * @return The log Gamma function: <i>ln(gamma(alpha))</i> for alpha > 0
	 */
	public static double computeLnGamma(double alpha)
	{
		if(alpha <= 0)
			throw new IllegalArgumentException("alpha must be grater than 0");
		
		double x = alpha;
		double f = 0.0;
		double z = Double.NaN;
	
	    if(x < 7) 
	    {
	        f = 1;
	        z = x;
	        while(z < 7)
	        {
	        	f*= z;
	        	z++;
	        }
	        x = z;
	        f = -Math.log(f);
	    }
	    z = 1/(x*x);
	
	    return f + (x - 0.5)*Math.log(x) - x + 0.918938533204673 + (((-0.000595238095238 * z + 
	    			0.000793650793651)*z - 0.002777777777778)*z + 0.083333333333333)/x;
	}

	/**
	 * Computes the value of a random variable with a chi-squared distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. The <i>p</i> parameter must be within the closed
	 * interval <i>[0, 1]</i>.
	 * @param k The number of degrees of freedom of the Chi-squared distribution
	 * @param p The value of the cumulative distribution function
	 * @return The inverse cumulative distribution function value
	 */
	private static double pointChi2(double k, double p)
	{
		final double e = 0.5e-6;
		final double aa = 0.6931471805; 
		final double prob = p;
	    double ch, a, q, p1, p2, t, x, b, s1, s2, s3, s4, s5, s6;
	    double epsi = (prob < 0.000002 || prob > 1 - 0.000002) ? 0.000001 : 0.01;
	    double g = computeLnGamma(k/2);
	    double xx = k/2;
	    double c = xx - 1;
	    if(k < -1.24 * Math.log(prob)) 
	    {
	        ch = Math.pow((prob*xx*Math.exp(g + xx*aa)), 1/xx);
	        if(ch - e < 0)
	            return ch;
	    }
	    else 
	    {
	        if(k > 0.32)
	        {
	            x = Normal.computeInvCDF(prob);
	            p1 = 0.222222/k;
	            ch = k*Math.pow((x*Math.sqrt(p1) + 1 - p1), 3.0);
	            if (ch > 2.2*k + 6)
	                ch = -2*(Math.log(1 - prob) - c*Math.log(0.5*ch) + g);
	        } 
	        else 
	        {
	            ch = 0.4;
	            a = Math.log(1 - prob);
	            do 
	            {
	                q = ch;
	                p1 = 1 + ch*(4.67 + ch);
	                p2 = ch*(6.73 + ch*(6.66 + ch));
	                t = -0.5 + (4.67 + 2*ch)/p1 - (6.73 + ch*(13.32 + 3*ch))/p2;
	                ch-= (1 - Math.exp(a + g + 0.5*ch + c*aa)*p2/p1)/t;
	            } while(Math.abs(q/ch - 1) - epsi > 0);
	        }
	    }
	    do 
	    {
	        q = ch;
	        p1 = 0.5*ch;
	        t = incompleteGammaFunctionP(xx, p1, g);
	        if(t < 0)
	            throw new IllegalArgumentException("arguments out of range: t < 0");
	        p2 = prob - t;
	        t = p2*Math.exp(xx*aa + g + p1 - c*Math.log(ch));
	        b = t/ch;
	        a = 0.5*t - b*c;
	
	        s1 = (210 + a*(140 + a*(105 + a*(84 + a*(70 + 60*a)))))/420;
	        s2 = (420 + a*(735 + a*(966 + a*(1141 + 1278*a))))/2520;
	        s3 = (210 + a*(462 + a*(707 + 932*a)))/2520;
	        s4 = (252 + a*(672 + 1182*a) + c*(294 + a*(889 + 1740*a)))/5040;
	        s5 = (84 + 264*a + c*(175 + 606*a))/2520;
	        s6 = (120 + c*(346 + 127*c))/5040;
	        ch += t*(1 + 0.5*t*s1 - b*c*(s1 - b*(s2 - b*(s3 - b*(s4 - b*(s5 - b*s6))))));
	    } while(Math.abs(q/ch - 1) > e);
	
	    return (ch);
	}

	/**
	 * Computes the incomplete gamma ratio <i>I(x,alpha)</i> where <i>x</i> is the upper limit of 
	 * the integration and <i>alpha</i> is the shape parameter
	 * @param alpha Parameter of the Gamma function 
	 * @param x Upper limit of integration
	 * @return The incomplete gamma ratio <i>I(x,alpha)</i>
	 */
	private static double incompleteGammaFunctionP(double alpha, double x) 
	{	
		double lnGammaAlpha = computeLnGamma(alpha);
		double accurate = 1e-8;
		double overflow = 1e30;
        double factor, gin, rn, a, b, an, dif, term;
        double pn0, pn1, pn2, pn3, pn4, pn5;

        if (x == 0.0)
            return 0.0;
        
        if (x < 0.0 || alpha <= 0.0)
            throw new IllegalArgumentException("arguments are out of bounds");

        factor = Math.exp(alpha * Math.log(x) - x - lnGammaAlpha);

        if(x > 1 && x >= alpha) 
        {
            // continued fraction
            a = 1 - alpha;
            b = a + x + 1;
            term = 0;
            pn0 = 1;
            pn1 = x;
            pn2 = x + 1;
            pn3 = x*b;
            gin = pn2/pn3;

            do 
            {
                a++;
                b+= 2;
                term++;
                an = a*term;
                pn4 = b*pn2 - an*pn0;
                pn5 = b*pn3 - an*pn1;

                if(pn5 != 0) 
                {
                    rn = pn4/pn5;
                    dif = Math.abs(gin - rn);
                    if(dif <= accurate)
                        if(dif <= accurate*rn)
                            break;
                    gin = rn;
                }
                pn0 = pn2;
                pn1 = pn3;
                pn2 = pn4;
                pn3 = pn5;
                if(Math.abs(pn4) >= overflow) 
                {
                    pn0/= overflow;
                    pn1/= overflow;
                    pn2/= overflow;
                    pn3/= overflow;
                }
            } while(true);
            gin = 1 - factor*gin;
        } 
        else 
        {
            // series expansion
            gin = 1;
            term = 1;
            rn = alpha;
            do 
            {
                rn++;
                term*= x/rn;
                gin+= term;
            }
            while(term > accurate);
            gin*= factor / alpha;
        }
        return gin;
	}
	
	/**
	 * Computes the incomplete gamma ratio <i>I(x,alpha)</i> where <i>x</i> is the upper limit of 
	 * the integration and <i>alpha</i> is the shape parameter
	 * @param alpha Parameter of the Gamma function 
	 * @param x Upper limit of integration
	 * @param lnGammaAlpha The log Gamma function for alpha
	 * @return The incomplete gamma ratio <i>I(x,alpha)</i>
	 */
	private static double incompleteGammaFunctionP(double alpha, double x, double lnGammaAlpha) 
	{
		double accurate = 1e-8;
		double overflow = 1e30;
        double factor, gin, rn, a, b, an, dif, term;
        double pn0, pn1, pn2, pn3, pn4, pn5;

        if (x == 0.0)
            return 0.0;
        
        if (x < 0.0 || alpha <= 0.0)
            throw new IllegalArgumentException("arguments are out of bounds");

        factor = Math.exp(alpha * Math.log(x) - x - lnGammaAlpha);

        if(x > 1 && x >= alpha) 
        {
            // continued fraction
            a = 1 - alpha;
            b = a + x + 1;
            term = 0;
            pn0 = 1;
            pn1 = x;
            pn2 = x + 1;
            pn3 = x*b;
            gin = pn2/pn3;

            do 
            {
                a++;
                b+= 2;
                term++;
                an = a*term;
                pn4 = b*pn2 - an*pn0;
                pn5 = b*pn3 - an*pn1;

                if(pn5 != 0) 
                {
                    rn = pn4/pn5;
                    dif = Math.abs(gin - rn);
                    if(dif <= accurate)
                        if(dif <= accurate*rn)
                            break;
                    gin = rn;
                }
                pn0 = pn2;
                pn1 = pn3;
                pn2 = pn4;
                pn3 = pn5;
                if(Math.abs(pn4) >= overflow) 
                {
                    pn0/= overflow;
                    pn1/= overflow;
                    pn2/= overflow;
                    pn3/= overflow;
                }
            } while(true);
            gin = 1 - factor*gin;
        } 
        else 
        {
            // series expansion
            gin = 1;
            term = 1;
            rn = alpha;
            do 
            {
                rn++;
                term*= x/rn;
                gin+= term;
            }
            while(term > accurate);
            gin*= factor / alpha;
        }
        return gin;
	}
	
}
