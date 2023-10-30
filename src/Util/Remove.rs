use arrayfire;
use half;

use crate::Util::Search::find_unique;
use std::collections::HashMap;



const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;





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
















pub fn select_forward_sphere<Z: arrayfire::FloatingPoint>(
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







    let colseq = arrayfire::Seq::new(0.0, (space_dims-1) as f64, 1.0);

    let mut temparr = arrayfire::constant::<f64>(0.0,arrayfire::Dim4::new(&[neuron_size,space_dims,1,1]));

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);









    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    let row_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

	let mut row_magsq = arrayfire::pow(&row_neuron_pos,&two,false);
	row_magsq = arrayfire::sum(&row_magsq, 1);







    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WColIdx, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    let col_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

	let mut col_magsq = arrayfire::pow(&col_neuron_pos,&two,false);
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
















