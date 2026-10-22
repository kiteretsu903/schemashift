"""Serialize the pinned SDK's dataclass forward/backward result explicitly."""
from __future__ import annotations


def forward_backward_record(output):
    """TensorData is also a dataclass with an internal ndarray, not a Pydantic model."""
    records=[]
    for row in output.loss_fn_outputs:
        record={}
        for name,tensor in row.items():
            record[name]={'data':list(tensor.data),'dtype':tensor.dtype,'shape':tensor.shape,
                'sparse_crow_indices':tensor.sparse_crow_indices,
                'sparse_col_indices':tensor.sparse_col_indices}
        records.append(record)
    return {'loss_fn_output_type':output.loss_fn_output_type,
        'loss_fn_outputs':records,'metrics':dict(output.metrics)}
