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

import probDist.multiVar.MultiVarKernelDensity;

/**
 * This interface allows accessing an array of continuous values to be used as a sample of a
 * kernel density distribution
 * @see MultiVarKernelDensity
 * @author Felipe Hernández
 */
public interface ContMultiSample 
{
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return The relative weight of the sample. A constant or empty value can be used if the 
	 * samples have uniform weights.
	 */
	public double getWeight();
	
	/**
	 * @param weight The relative weight of the sample. A constant or empty value can be used if 
	 * the samples have uniform weights.
	 */
	public void setWeight(double weight);
	
	/**
	 * @return The list of values in the sample. The order of the vales should be consistent
	 * between different samples. The list can include missing values that should be represented as
	 * {@link java.lang.Double#NaN}.
	 */
	public ArrayList<Double> getValues();

	/**
	 * @return A new object with the same attributes as the original.
	 */
	public ContMultiSample copy();
	
}
