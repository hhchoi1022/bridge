from fastavro import writer, parse_schema
import numpy as np
import json
import math
from pathlib import Path


class AvroWriter:

    def __init__(self):
        pass

    def _infer_field_type(self, values):

        non_null = []

        for v in values:
            if v is None:
                continue
            if isinstance(v, np.ma.core.MaskedConstant):
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            non_null.append(v)

        if len(non_null) == 0:
            return ["null", "string"]

        has_bool = False
        has_int = False
        has_float = False
        has_str = False

        for v in non_null:

            if isinstance(v, (bool, np.bool_)):
                has_bool = True

            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                has_int = True

            elif isinstance(v, (float, np.floating)):
                has_float = True

            elif isinstance(v, str):
                has_str = True

            else:
                return ["null", "string"]

        if (has_int or has_float) and not has_str and not has_bool:
            if has_float:
                return ["null", "double"]
            return ["null", "long"]

        if has_bool and not (has_int or has_float or has_str):
            return ["null", "boolean"]

        return ["null", "string"]

    def make_schema(self, tbl):

        fields = []

        for col in tbl.colnames:

            field_type = self._infer_field_type(tbl[col])

            fields.append({
                "name": col,
                "type": field_type
            })

        schema = {
            "type": "record",
            "name": "alert",
            "fields": fields
        }

        return parse_schema(schema)

    def normalize_value(self, val, field_type):

        if val is None:
            return None

        if isinstance(val, np.ma.core.MaskedConstant):
            return None

        if isinstance(val, np.generic):
            val = val.item()

        if isinstance(val, np.ndarray):
            val = val.tolist()

        if isinstance(val, float):
            if not math.isfinite(val):
                return None

        avro_type = field_type[-1]

        if avro_type == "double":

            try:
                return float(val)
            except:
                return None

        elif avro_type == "long":

            try:
                return int(val)
            except:
                return None

        elif avro_type == "boolean":

            if isinstance(val, (bool, np.bool_)):
                return bool(val)

            return None

        elif avro_type == "string":

            if isinstance(val, (dict, list, tuple)):
                return json.dumps(val)

            return str(val)

        return None

    def write_table(self, tbl, filename, schema=None):

        if schema is None:
            schema = self.make_schema(tbl)

        field_types = {f["name"]: f["type"] for f in schema["fields"]}

        record = {
            col: self.normalize_value(tbl[col], field_types[col])
            for col in tbl.colnames
        }

        with open(filename, "wb") as out:
            writer(out, schema, [record])

        return schema