use arrayfire;
use half;

use crate::Util::Search::find_unique;
use crate::Util::Search::COO_batch_find;
use std::collections::HashMap;



const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;

const COO_FIND_LIMIT: u64 = 1500000000;



pub fn select_values<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    sel: &arrayfire::Array<u32>
)
{

    *WValues = arrayfire::lookup(WValues, sel, 0);
    *WRowIdxCOO = arrayfire::lookup(WRowIdxCOO, sel, 0);
    *WColIdx = arrayfire::lookup(WColIdx, sel, 0);

}




pub fn clear_input<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    input_rows: u64
)
{

    let single = input_rows as i32;

    
    let cmp1 = arrayfire::le(&single ,WRowIdxCOO, false);

    let sel = arrayfire::locate(&cmp1);

    select_values::<Z>(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &sel
    );

}





pub fn clear_output<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    output_cols: u64
)
{

    let single = output_cols as i32;

    
    let cmp1 = arrayfire::lt(WColIdx, &single, false);

    let sel = arrayfire::locate(&cmp1);

    select_values::<Z>(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &sel
    );
}











pub fn clear_input_to_hidden<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    input_cols: u64
)
{


    let single = input_cols as i32;

    
    let cmp1 = arrayfire::ge(WColIdx, &single, false);

    let sel = arrayfire::locate(&cmp1);

    select_values::<Z>(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &sel
    );
}














pub fn delete_smallest_weights<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    del_num: u64
)
{
    let WValues_num  = WValues.dims()[0];

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut idx = arrayfire::constant::<u32>(0,single_dims);

    if WValues.is_half()
    {
        let abs = arrayfire::abs(&WValues).cast::<half::f16>();
        (_,idx) = arrayfire::sort_index(&abs, 0, false);
    }
    else if WValues.is_single()
    {
        let abs = arrayfire::abs(&WValues).cast::<f32>();
        (_,idx) = arrayfire::sort_index(&abs, 0, false);
    }
    else 
    {
        let abs = arrayfire::abs(&WValues).cast::<f64>();
        (_,idx) = arrayfire::sort_index(&abs, 0, false);
    }





    //Select biggest weights
    let mut sel = arrayfire::rows(&idx, 0, (WValues_num-del_num-1)  as i64);

    sel = find_unique(
        &sel,
        WValues_num
    );



    //Select COO Matrix
    select_values(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &sel
    );
}







pub fn delete_weights_with_prob<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    del_num: u64
)
{

    let WValues_num  = WValues.dims()[0];
    





    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut idx = arrayfire::constant::<u32>(0,single_dims);

    if WValues.is_half()
    {
        let mut abs = arrayfire::abs(&WValues).cast::<half::f16>();
        let randarr = arrayfire::randu::<half::f16>(abs.dims());
        abs = arrayfire::mul(&abs, &randarr, false);
        (_,idx) = arrayfire::sort_index(&abs, 0, false);
    }
    else if WValues.is_single()
    {
        let mut abs = arrayfire::abs(&WValues).cast::<f32>();
        let randarr = arrayfire::randu::<f32>(abs.dims());
        abs = arrayfire::mul(&abs, &randarr, false);
        (_,idx) = arrayfire::sort_index(&abs, 0, false);
    }
    else 
    {
        let mut abs = arrayfire::abs(&WValues).cast::<f64>();
        let randarr = arrayfire::randu::<f64>(abs.dims());
        abs = arrayfire::mul(&abs, &randarr, false);
        (_,idx) = arrayfire::sort_index(&abs, 0, false);
    }





    //Select biggest weights
    let mut sel = arrayfire::rows(&idx, 0, (WValues_num-del_num-1)  as i64);

    sel = find_unique(
        &sel,
        WValues_num
    );



    //Select COO Matrix
    select_values(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &sel
    );





}
















pub fn select_forward_sphere<Z: arrayfire::FloatingPoint<AggregateOutType = Z> >(
    modeldata_int: &HashMap<String, u64>,

    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    neuron_pos: &arrayfire::Array<Z>,
    neuron_idx: &arrayfire::Array<i32>
){

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();


    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = modeldata_int["input_size"].clone();
    let output_size: u64 = modeldata_int["output_size"].clone();
    let space_dims: u64 = modeldata_int["space_dims"].clone();







    let colseq = arrayfire::Seq::new(0, (space_dims-1) as i32, 1);

    let mut temparr = arrayfire::constant::<f64>(ZERO_F64,arrayfire::Dim4::new(&[neuron_size,space_dims,1,1])).cast::<Z>();

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);









    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    let row_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

	let mut row_magsq = arrayfire::pow(&row_neuron_pos,&TWO,false);
	row_magsq = arrayfire::sum(&row_magsq, 1);







    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WColIdx, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    let col_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

	let mut col_magsq = arrayfire::pow(&col_neuron_pos,&TWO,false);
	col_magsq = arrayfire::sum(&col_magsq, 1);






    //let cmp1 = (WRowIdxCOO < WColIdx);
    let cmp1 = arrayfire::lt(&row_magsq ,&col_magsq, false);

	let sel = arrayfire::locate(&cmp1);

    select_values(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &sel
    );

}















pub fn delete_neurons_at_idx<Z: arrayfire::FloatingPoint >(
    delete_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>
){


    let COO_batch_size = 1 + ((COO_FIND_LIMIT/WColIdx.dims()[0]) as u64);
    let valsel = COO_batch_find(WColIdx,&delete_idx, COO_batch_size);





    let mut temparr = arrayfire::constant::<bool>(true,WColIdx.dims());

    let ones = arrayfire::constant::<bool>(false,valsel.dims());

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&valsel, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);





    let valsel2 = arrayfire::locate(&temparr);

    select_values(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &valsel2
    );

}









pub fn delete_unused_neurons<Z: arrayfire::FloatingPoint>(
    modeldata_int: &HashMap<String, u64>,


    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    glia_pos: &mut arrayfire::Array<Z>,
    neuron_pos: &mut arrayfire::Array<Z>,
    neuron_idx: &mut arrayfire::Array<i32>
){

    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = modeldata_int["input_size"].clone();
    let output_size: u64 = modeldata_int["output_size"].clone();
    let space_dims: u64 = modeldata_int["space_dims"].clone();

    

    //Get active non zero cols
    let mut temparr = arrayfire::constant::<bool>(false,arrayfire::Dim4::new(&[neuron_size,1,1,1]));

    let ones = arrayfire::constant::<bool>(true,WColIdx.dims());

    let idx = WColIdx.clone();
    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);






    //Get all non zero index of col
    let mut sel = arrayfire::locate(&temparr).cast::<i32>();

    let active_size = neuron_idx.dims()[0];
    let output_idx = arrayfire::rows(neuron_idx, (active_size-output_size)  as i64, (active_size-1)   as i64);
    //Add output neurons to index
    sel = arrayfire::join(0, &sel, &output_idx);



    sel = find_unique(
        &sel,
        neuron_size
    );



    let update_neuron_idx = sel.clone();






    let COO_batch_size = 1 + ((COO_FIND_LIMIT/WRowIdxCOO.dims()[0]) as u64);

    let valsel = COO_batch_find(WRowIdxCOO,&sel, COO_batch_size).cast::<u32>();

    if (valsel.dims()[0] == WRowIdxCOO.dims()[0])
    {
        return;
    }








    select_values(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &valsel
    );


    let mut temparr = arrayfire::constant::<f64>(ZERO_F64,arrayfire::Dim4::new(&[neuron_size,space_dims,1,1])).cast::<Z>();

    let seq1 = arrayfire::Seq::new(0, (space_dims-1) as i32, 1);
    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);










    let mut temparr2 = arrayfire::constant::<bool>(false,arrayfire::Dim4::new(&[neuron_size,1,1,1]));
    let ones = arrayfire::constant::<bool>(true,idx.dims());

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    arrayfire::assign_gen(&mut temparr2, &idxrs, &ones);



    let zeros = arrayfire::constant::<bool>(false,sel.dims());
    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    arrayfire::assign_gen(&mut temparr2, &idxrs, &zeros);

    let new_glia_idx = arrayfire::locate(&temparr2);
    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&new_glia_idx, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));
    let new_glia_pos = arrayfire::index_gen(&temparr, idxrs);

    *glia_pos = arrayfire::join(0, glia_pos, &new_glia_pos);







    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));
    *neuron_pos = arrayfire::index_gen(&temparr, idxrs);




    *neuron_idx = update_neuron_idx;
}





