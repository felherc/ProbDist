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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import utilities.Utilities;
import utilities.stat.ContSeries;

/**
 * This class represents multinomial probability distributions for discrete random variables
 * @author Felipe Hernández
 */
public class DiscProbDist 
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Undefined distribution type identifier
	 */
	public final static int NONE = 0;
	
	/**
	 * Uniform distribution type identifier: all possible values have the same probability
	 */
	public final static int UNIFORM = 1;
	
	/**
	 * Look-up distribution type identifier: each value has a probability assigned to it
	 */
	public final static int LOOKUP = 2;
	
	/**
	 * Continuous distribution type identifier: values are rounded from a continuous probability
	 * distribution
	 */
	public final static int CONTINUOUS = 3;

	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Identifier of the type of distribution
	 */
	private int type;
	
	/**
	 * The minimum value of the distribution
	 */
	private int min;
	
	/**
	 * The maximum value of the distribution
	 */
	private int max;
	
	/**
	 * The probability array for the values of the distribution
	 */
	private ArrayList<Double> probs;
	
	/**
	 * The sum of the probabilities of the values in the distribution
	 */
	private double probSum;
	
	/**
	 * The continuous probability distribution to round values from
	 */
	private ContProbDist continuous;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a discrete probability distribution
	 */
	public DiscProbDist()
	{
		type = NONE;
		min = 0;
		max = 0;
		probs = null;
		probSum = Double.NaN;
		continuous = null;
	}
	
	/**
	 * Creates a discrete probability distribution with uniform probabilities
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 */
	public DiscProbDist(int min, int max)
	{
		type = UNIFORM;
		this.min = min <= max ? min : max;
		this.max = max >= min ? max : min;
		probs = null;
		probSum = Double.NaN;
		continuous = null;
	}
	
	/**
	 * Creates a discrete probability distribution where probabilities are obtained using a look-up 
	 * table
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param probs The array with the probabilities of each of the values of the distribution
	 */
	public DiscProbDist(int min, int max, Collection<Double> probs)
	{
		type = LOOKUP;
		this.min = min <= max ? min : max;
		this.max = max >= min ? max : min;
		this.probs = new ArrayList<Double>();
		Iterator<Double> iter = probs.iterator();
		probSum = 0;
		for(int i = 0 ; i < (this.max - this.min + 1) ; i++)
		{
			double value;
			if(i < probs.size())
				value = Math.max(0, iter.next());
			else
				value = 0;
			this.probs.add(value);
			probSum += value;
		}
		continuous = null;
	}
	
	/**
	 * Creates a discrete probability distribution where values are rounded from a continuous 
	 * probability distribution
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param contDist The continuous probability distribution to round values from
	 */
	public DiscProbDist(int min, int max, ContProbDist contDist)
	{
		type = CONTINUOUS;
		this.min = min <= max ? min : max;
		this.max = max >= min ? max : min;
		probs = null;
		probSum = Double.NaN;
		continuous = contDist;
	}
	
	/**
	 * Creates a discrete probability distribution where probabilities are obtained using a look-up 
	 * table which is created from a series of observed values
	 * @param values The sample values to create the distribution from
	 */
	public DiscProbDist(Collection<Integer> values)
	{
		ContSeries series = new ContSeries();
		for(Integer value : values)
			series.addValue(value);
		adjustProbDist(series);
	}
	
	/**
	 * Creates a discrete probability distribution where probabilities are obtained using a look-up 
	 * table which is created from a series of observed values
	 * @param series The sample values to create the distribution from
	 */
	public DiscProbDist(ContSeries series)
	{
		adjustProbDist(series);
	}
	
	/**
	 * Creates a discrete probability distribution where values are rounded from a continuous 
	 * probability distribution. The continuous distribution is created from a series of observed 
	 * values
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param series The sample values to create the distribution from
	 */
	public DiscProbDist(int min, int max, ContSeries series)
	{
		adjustProbDist(min, max, series);
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a discrete probability distribution where probabilities are obtained using a look-up 
	 * table which is created from a series of observed values
	 * @param series The sample values to create the distribution from
	 */
	public void adjustProbDist(ContSeries series)
	{
		type = LOOKUP;
		min = (int) series.getMin();
		max = (int) series.getMax();
		probs = new ArrayList<Double>();
		for(int i = 0 ; i <= (max - min) ; i++)
			probs.add(0.0);
		ArrayList<Double> values = series.getValues();
		if(series.isWeighted())
		{
			probSum = 0;
			ArrayList<Double> weights = series.getWeightingFactors();
			for(int i = 0 ; i < values.size() ; i++)
			{
				int value = (int)(double)values.get(i);
				double weight = weights.get(i);
				probs.set(value - min, weight);
				probSum += weight;
			}
		}
		else
		{
			for(Double value : values)
			{
				int index = ((int)(double) value) - min;
				probs.set(index, probs.get(index) + 1);
			}
			probSum = values.size();
		}
		continuous = null;
	}
	
	/**
	 * Creates a discrete probability distribution where values are rounded from a continuous 
	 * probability distribution. The continuous distribution is created from a series of observed 
	 * values
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param series The sample values to create the distribution from
	 * @return The root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of the continuous 
	 * distribution.
	 */
	public void adjustProbDist(int min, int max, ContSeries series)
	{
		type = CONTINUOUS;
		this.min = min <= max ? min : max;
		this.max = max >= min ? max : min;
		probs = null;
		probSum = Double.NaN;
		continuous = ContProbDist.fromValues(series);
	}

	/**
	 * @return Identifier of the type of distribution. The different types are defined as class
	 * constants
	 */
	public int getType() 
	{
		return type;
	}

	/**
	 * @return The minimum value of the distribution
	 */
	public int getMin() 
	{
		return min;
	}

	/**
	 * @return The maximum value of the distribution
	 */
	public int getMax() 
	{
		return max;
	}

	/**
	 * @return The probability array for the values of the distribution. Returns null if the 
	 * distribution type is not LOOKUP.
	 */
	public ArrayList<Double> getProbs() 
	{
		return probs;
	}

	/**
	 * @return The sum of the probabilities of the values in the distribution. Returns Double.NaN 
	 * if the distribution type is not LOOKUP.
	 */
	public double getProbSum() 
	{
		return probSum;
	}

	/**
	 * @return The continuous probability distribution to round values from. Returns null if the 
	 * distribution type is not CONTINUOUS.
	 */
	public ContProbDist getContinuous() 
	{
		return continuous;
	}
	
	/**
	 * Computes the probability of a random variable with this distribution of having the provided
	 * value. Returns Double.NaN if the distribution has not been correctly defined.
	 * @param x The value to sample
	 * @return The probability of a random variable with this distribution of having the provided
	 * value
	 */
	public double getProb(int x)
	{
		return getProb(x, x);
	}
	
	/**
	 * Computes the probability of a random variable with this distribution of falling inside the 
	 * provided closed interval. Returns Double.NaN if the distribution has not been correctly 
	 * defined.
	 * @param lower The lower limit of the sampling closed interval
	 * @param upper The upper limit of the sampling closed interval
	 * @return The probability of a random variable with this distribution of falling inside the 
	 * provided closed interval
	 */
	public double getProb(int lower, int upper)
	{
		lower = lower <= upper ? lower : upper;
		upper = upper >= lower ? upper : lower;
		if(lower > max || upper < min)
			return 0;
		lower = lower < min ? min : lower;
		upper = upper > max ? max : upper;
		switch(type)
		{
			case UNIFORM: 		double range = upper - lower + 1;
								double totalRange = max - min + 1;
								return range / totalRange;
			case LOOKUP:		double sum = 0;
								for(int i = lower - min ; i <= (upper - min) ; i++)
									sum += probs.get(i)/probSum;
								return sum;
			case CONTINUOUS:	double inf = lower == min ? Double.NEGATIVE_INFINITY : lower - 0.5;
								double sup = upper == max ? Double.POSITIVE_INFINITY : upper + 0.5;
								return continuous.getProb(inf, sup);
			default:			return Double.NaN; 
		}
	}
	
	/**
	 * Computes the probability of a random variable with this distribution of being equal to or 
	 * greater than the provided value. Returns Double.NaN if the distribution has not been 
	 * correctly defined.
	 * @param x The inferior limit of the sampling interval
	 * @return The probability of a random variable with this distribution of being equal to or 
	 * greater than the provided value
	 */
	public double getSupProb(int x)
	{
		if(x > max)
			return 0;
		return getProb(x, max);
	}
	
	/**
	 * Computes the probability of a random variable with this distribution of being equal to or 
	 * smaller than the provided value. Returns Double.NaN if the distribution has not been 
	 * correctly defined.
	 * @param x The superior limit of the sampling interval
	 * @return The probability of a random variable with this distribution of being equal to or 
	 * smaller than the provided value
	 */
	public double getInfProb(int x)
	{
		if(x < min)
			return 0;
		return getProb(min, x);
	}
	
	/**
	 * Computes the expected value of a random variable with this distribution constrained to the 
	 * interval between the two parameter values. Returns Double.NaN if the distribution has not 
	 * been correctly defined.
	 * @param lower The lower limit of the sampling closed interval
	 * @param upper The upper limit of the sampling closed interval
	 * @return The expected value of a random variable with this distribution constrained to the 
	 * interval between the two parameter values
	 */
	public double getExpectation(int lower, int upper)
	{
		lower = lower <= upper ? lower : upper;
		upper = upper >= lower ? upper : lower;
		if(lower > max || upper < min)
			return Double.NaN;
		lower = lower < min ? min : lower;
		upper = upper > max ? max : upper;
		double inf = Double.NaN;
		double sup = Double.NaN;
		switch(type)
		{
			case UNIFORM: 		inf = lower;
								sup = upper;
								return (inf + sup)/2;
			case LOOKUP:		ContSeries series = new ContSeries(true);
								for(int i = lower - min ; i < upper - min ; i++)
									series.addValue(i + min, probs.get(i));
								return series.getMean();
			case CONTINUOUS:	inf = lower == min ? Double.NEGATIVE_INFINITY : lower - 0.5;
								sup = upper == max ? Double.POSITIVE_INFINITY : upper + 0.5;
								ContProbDist truncated = continuous.truncate(inf, sup); 
								return truncated.getMean();
			default:			return Double.NaN;
		}
	}
	
	/**
	 * Computes the expected value of a random variable with this distribution constrained to be 
	 * equal to or larger than the parameter value. Returns Double.NaN if the parameter value is 
	 * greater than the maximum value of the distribution or if the distribution has not been 
	 * correctly defined.
	 * @param x The lower limit of the random variable
	 * @return The expected value of a random variable with this distribution constrained to be 
	 * equal to or larger than the parameter value
	 */
	public double getSupExpectation(int x)
	{
		if(x > max)
			return Double.NaN;
		return getExpectation(x, max);
	}
	
	/**
	 * Computes the expected value of a random variable with this distribution constrained to be 
	 * equal to or smaller than the parameter value. Returns Double.NaN if the parameter value is 
	 * smaller than the minimum value of the distribution or if the distribution has not been 
	 * correctly defined.
	 * @param x The upper limit of the random variable
	 * @return The expected value of a random variable with this distribution constrained to be 
	 * equal to or smaller than the parameter value
	 */
	public double getInfExpectation(int x)
	{
		if(x < min)
			return Double.NaN;
		return getExpectation(min, x);
	}
	
	/**
	 * Generates a random number with this distribution. Returns Integer.Min_VALUE if the 
	 * distribution has not been correctly defined.
	 * @return A random number with this distribution
	 */
	public int sample()
	{
		switch(type)
		{
			case UNIFORM: 		return min + Utilities.uniformRandomSelect(max - min + 1);
			case LOOKUP:		if (probs.size() == 0)
									return Integer.MIN_VALUE;
								if (probs.size() == 1)
									return min;
								double random = Math.random()*probSum;
								double sum = 0;
								int index = 0;
								while(sum <= random && index < probs.size())
								{
									sum += probs.get(index);
									index++;
								}
								return index - 1;
			case CONTINUOUS:	int discRnd = Utilities.round(continuous.sample());
								discRnd = discRnd < min ? min : discRnd;
								discRnd = discRnd > max ? max : discRnd;
								return discRnd;
			default:			return Integer.MIN_VALUE;
		}
	}
	
}