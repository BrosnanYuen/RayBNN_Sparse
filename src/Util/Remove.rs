use arrayfire;

use crate::Util::Search::find_unique;


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

    //let cmp2 = (single <= WRowIdxCOO );
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

    //let cmp2 = (WColIdx < single );
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

    //let cmp2 = (WColIdx >= single );
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
    let abs = arrayfire::abs(&WValues);
    //Sort to find small weights
    let (_,mut idx) = arrayfire::sort_index(&abs, 0, false);



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









