use arrayfire;
use half;

use crate::Util::Convert::get_global_weight_idx;

use std::collections::HashMap;

use nohash_hasher;

use rand::distributions::{Distribution, Uniform};


use rand::seq::SliceRandom;

use rayon::prelude::*;



pub fn add_random_weights<Z: arrayfire::FloatingPoint>(
    modeldata_int: &HashMap<String, u64>,

    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,

    add_num: u64
)
{

    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = modeldata_int["input_size"].clone();
    let output_size: u64 = modeldata_int["output_size"].clone();
    let proc_num: u64 = modeldata_int["proc_num"].clone();
    let active_size: u64 = modeldata_int["active_size"].clone();
    let space_dims: u64 = modeldata_int["space_dims"].clone();
    let step_num: u64 = modeldata_int["step_num"].clone();
    let batch_size: u64 = modeldata_int["batch_size"].clone();

    

	//Compute global index
	let mut gidx1 = get_global_weight_idx(
		neuron_size,
		&WRowIdxCOO,
		&WColIdx,
	);


	let mut gidx1_cpu = vec!(u64::default();gidx1.elements());
    gidx1.host(&mut gidx1_cpu);



	let mut WValues_cpu = vec!(Z::default();WValues.elements());
    WValues.host(&mut WValues_cpu);

	let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

	let mut WColIdx_cpu = vec!(i32::default();WColIdx.elements());
    WColIdx.host(&mut WColIdx_cpu);


    let neuron_dims = neuron_idx.dims();
	let neuron_num = neuron_dims[0];
    
	let hidden_idx = arrayfire::rows(neuron_idx, input_size as i64, (neuron_num-output_size-1)  as i64);

    let input_idx = arrayfire::rows(&neuron_idx, 0, (input_size-1)  as i64);

	let output_idx = arrayfire::rows(&neuron_idx, (neuron_idx.dims()[0]-output_size) as i64, (neuron_idx.dims()[0]-1)  as i64);

    let mut hidden_idx_cpu = vec!(i32::default();hidden_idx.elements());
    hidden_idx.host(&mut  hidden_idx_cpu);

    let mut input_idx_cpu = vec!(i32::default();input_idx.elements());
    input_idx.host(&mut  input_idx_cpu);

    let mut output_idx_cpu = vec!(i32::default();output_idx.elements());
    output_idx.host(&mut  output_idx_cpu);




	let mut join_WValues = nohash_hasher::IntMap::default();
	let mut join_WColIdx = nohash_hasher::IntMap::default();
	let mut join_WRowIdxCOO = nohash_hasher::IntMap::default();

    for qq in 0..gidx1.elements()
	{
		let cur_gidx = gidx1_cpu[qq].clone();

		join_WValues.insert(cur_gidx, WValues_cpu[qq].clone());
		join_WColIdx.insert(cur_gidx, WColIdx_cpu[qq].clone());
		join_WRowIdxCOO.insert(cur_gidx, WRowIdxCOO_cpu[qq].clone());
	}

    let mut rng = rand::thread_rng();
    let choose_connection = Uniform::from(0.0..1.0f64);
    let value_range = Uniform::from(-min_val..min_val);

    let p1 = (input_size as f64)/(neuron_num as f64);
    let p2 = ((input_size + hidden_idx.dims()[0]) as f64)/(neuron_num as f64);

    let mut add_counter = 0;
    while 1==1
    {
        let connection_type = choose_connection.sample(&mut rng);

        let mut cur_rows = 0;
        let mut cur_cols = 0;

        
        if connection_type <= p1
        {
            //Input to Hidden
            cur_rows = hidden_idx_cpu.choose(&mut rand::thread_rng()).unwrap().clone();
            cur_cols = input_idx_cpu.choose(&mut rand::thread_rng()).unwrap().clone();
        }
        else if (p1 < connection_type)  && (connection_type <= p2)
        {
            //Hidden to Hidden
            cur_rows = hidden_idx_cpu.choose(&mut rand::thread_rng()).unwrap().clone();
            cur_cols = hidden_idx_cpu.choose(&mut rand::thread_rng()).unwrap().clone();
        }
        else
        {
            //Hidden to Output
            cur_rows = output_idx_cpu.choose(&mut rand::thread_rng()).unwrap().clone();
            cur_cols = hidden_idx_cpu.choose(&mut rand::thread_rng()).unwrap().clone();
        }

        let cur_gidx = ((cur_rows as u64)*(neuron_size)) +  (cur_cols as u64);
        if join_WValues.contains_key(&cur_gidx) == false
        {
            let new_value = value_range.sample(&mut rng);
            join_WValues.insert(cur_gidx, new_value);
            join_WColIdx.insert(cur_gidx, cur_cols.clone());
		    join_WRowIdxCOO.insert(cur_gidx, cur_rows.clone());

            add_counter = add_counter + 1;

            if add_counter >= add_num
            {
                break;
            }
        }
    }


    let mut gidx3:Vec<u64> = join_WValues.clone().into_keys().collect();
	gidx3.par_sort_unstable();


	WValues_cpu = Vec::new();
	WRowIdxCOO_cpu = Vec::new();
	WColIdx_cpu = Vec::new();

	for qq in gidx3
	{
		WValues_cpu.push( join_WValues[&qq].clone() );
		WColIdx_cpu.push( join_WColIdx[&qq].clone() );
		WRowIdxCOO_cpu.push( join_WRowIdxCOO[&qq].clone() );
	}


	*WValues = arrayfire::Array::new(&WValues_cpu, arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]));
	*WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]));
	*WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]));
	


}





