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

package probDist.multiVar.tools;

import java.util.ArrayList;

/**
 * This class represents a sample of multi-dimensional continuous values
 * @author Felipe Hernández
 */
public class Sample implements ContMultiSample 
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Default value for {@link #weight}
	 */
	public final static double DEF_WEIGHT = 1.0;

	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Relative weight/importance value
	 */
	private double weight;
	
	/**
	 * List of the continuous values
	 */
	private ArrayList<Double> values;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Default constructor
	 */
	public Sample()
	{
		weight	= DEF_WEIGHT;
		values	= new ArrayList<>();
	}
	
	/**
	 * Constructor
	 * @param weight {@link #weight}
	 * @param values {@link #values}
	 */
	public Sample(double weight, ArrayList<Double> values)
	{
		this.weight	= weight;
		this.values	= values;
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	@Override
	public double getWeight() 
	{
		return weight;
	}

	@Override
	public void setWeight(double weight) 
	{
		this.weight = weight;
	}

	@Override
	public ArrayList<Double> getValues() 
	{
		return values;
	}

	/**
	 * @param values {@link #values}
	 */
	public void setValues(ArrayList<Double> values)
	{
		this.values = values;
	}
	
	/**
	 * Adds a value to {@link #values}
	 * @param value The value to add
	 */
	public void addValue(double value)
	{
		values.add(value);
	}
	
	public Sample copy()
	{
		ArrayList<Double> clonedValues = new ArrayList<Double>();
		for (Double value : values)
			clonedValues.add(new Double(value));
		return new Sample(weight, clonedValues);
	}
	
}
