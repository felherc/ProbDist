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

import utilities.Utilities;
import utilities.stat.ContSeries;
import utilities.stat.ContStats;

/**
 * This class represents mixture probability distributions for continuous random variables. The
 * mixture distribution is built from a series of component distributions, each with a given 
 * probability or weight.
 * @author Felipe Hernández
 */
public class Mixture extends ContProbDist 
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Mixture distribution type String identifier
	 */
	public final static String ID = "Mixture";
	
	/**
	 * Mixture distribution type short String identifier
	 */
	public final static String SHORT_ID = "Mix";
	
	/**
	 * pdf value identifier
	 */
	private final static int PDF = 1;
	
	/**
	 * CDF value identifier
	 */
	private final static int CDF = 2;
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The list of components of the mixture probability distribution
	 */
	protected ArrayList<ContProbDist> components;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a mixture probability distribution for a continuous random variable
	 * @param components The individual distributions to create the mixture distribution
	 */
	public Mixture(Collection<ContProbDist> components)
	{
		type 			= ContProbDist.MIXTURE;
		this.components = new ArrayList<ContProbDist>();
		this.components.addAll(components);
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return The list of the mixture components
	 */
	public ArrayList<ContProbDist> getComponents()
	{
		return components;
	}
	
	/**
	 * @param components The list of the mixture components
	 */
	public void setComponents(ArrayList<ContProbDist> components)
	{
		this.components = components;
	}
	
	@Override
	public String getTypeString() 
	{
		return ID;
	}

	@Override
	public String toString() 
	{
		String s = SHORT_ID + "{ ";
		for(ContProbDist component : components)
			s += component.getProb() + ": " + component.toString() + ", ";
		Utilities.removeLast(s);
		s += " }";
		return s;
	}

	@Override
	public String toString(int decimalPlaces) 
	{
		String s = SHORT_ID + "{ ";
		for(ContProbDist component : components)
		{
			double prob = Utilities.round(component.getProb(), decimalPlaces);
			s += prob + ": " + component.toString(decimalPlaces) + ", ";
		}
		Utilities.removeLast(s);
		s += " }";
		return s;
	}

	@Override
	public double getMean() 
	{
		ContSeries series = new ContSeries(true);
		for(ContProbDist component : components)
			series.addValue(component.getMean(), component.getProb());
		return series.size() > 0 ? series.getMean() : Double.NaN;
	}

	@Override
	public double getStDev() 
	{
		return Math.sqrt(getVar());
	}
	
	/**
	 * Computes the variance of the distribution using the law of total variance. The law of total
	 * variance requires adding the expected value of the conditional variances of each mixture 
	 * component distribution with the variance of the conditional means (which takes into account the 
	 * variance induced by the component distributions having different means). 
	 * @return The variance of the distribution
	 */
	public double getVar()
	{
		ContSeries series	= new ContSeries(true);
		double sum1			= 0;
		double sum2			= 0;
		double weightSum	= 0;
		for(ContProbDist component : components)
		{
			double weight	= component.getProb();
			double compMean	= component.getMean();
			
			series.addValue(component.getVar(), weight);
			sum1			+= weight*compMean;
			sum2			+= weight*compMean*compMean;
			weightSum		+= weight;
		}
		sum1				/= weightSum;
		sum2				/= weightSum;
		return series.getMean() + sum2 - sum1*sum1;
	}
	
	@Override
	public double getSkewness()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	@Override
	public double getpdf(double x) 
	{
		return computeValue(PDF, x);
	}
	
	@Override
	public double getCDF(double x) 
	{
		return computeValue(CDF, x);
	}

	@Override
	public double getInvCDF(double p) 
	{
		// TODO Needs to be computed iteratively (see KernelDensity)
		return Double.NaN;
	}
	
	@Override
	public ContProbDist truncate(double min, double max)
	{
		// TODO Implement
		return null;
	}

	/**
	 * Computes the weighted mean of a value from the mixture components
	 * @param valueCode The identifier of the value as defined by class constants
	 * @param parameter The parameter to compute the mean value
	 * @return The weighted mean of a value from the mixture components
	 */
	private double computeValue(int valueCode, double parameter)
	{
		ContSeries series = new ContSeries(true);
		for(ContProbDist component : components)
		{
			double value = Double.NaN;
			switch(valueCode)
			{
				case PDF:		value = component.getpdf(parameter); 	break;
				case CDF:		value = component.getCDF(parameter); 	break;
			}
			series.addValue(value, component.getProb());
		}
		return series.size() > 0 ? series.getMean() : Double.NaN;
	}

	@Override
	public double sample() 
	{
		double totalProb = 0;
		for(ContProbDist component : components)
			totalProb += component.getProb();
		if(totalProb <= 0)
			return Double.NaN;
		double selector = Math.random()*totalProb;
		ContProbDist selected = null;
		double sum = 0;
		int index = 0;
		while(sum <= selector)
		{
			selected = components.get(index);
			sum += selected.getProb();
			index++;
		}
		return selected.sample();
	}

	@Override
	public void shift(double newMean)
	{
		int cCount				= components.size();
		ArrayList<Double> means	= new ArrayList<>(cCount);
		ContStats stats			= new ContStats(true);
		for (ContProbDist component : components)
		{
			double mean			= component.getMean();
			means.add(mean);
			stats.addValue(mean, component.getProb());
		}
		double mean				= stats.getMean();
		double shift			= newMean - mean;
		for (int c = 0; c < cCount; c++)
		{
			ContProbDist comp	= components.get(c);
			comp.shift(means.get(c) + shift);
		}
	}

	@Override
	public void scale(double newStDev)
	{
		// TODO Implement
	}
	
	// TODO Implement a method to generate a series of random samples more efficiently: first 
	// compute the cumulative weight so that a binary search can be used.

}
