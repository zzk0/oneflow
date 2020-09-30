#include "{{ util.module_cfg_header_name(module) }}"

// {% for package in util.module_package_list(module) %}
// namespace {{ package }} {
// {% endfor %}
// namespace cfg {

// {% for cls in util.module_nested_message_types(module) %}
// {% if not util.class_is_map_entry(cls) %}
// {% for oneof in util.message_type_oneofs(cls) %}
// const _{{ util.class_name(cls) }}_::{{ util.oneof_enum_name(oneof) }} Const{{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET =
//                                   _{{ util.class_name(cls) }}_::{{ util.oneof_enum_name(oneof) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET;
// {% endfor %}{# oneof enum #}
// {% endif %}{# cls is not entry #}
// {% endfor %}{# cls #}

// }
// {% for package in util.module_package_list(module) %}
// } // namespace {{ package }}
// {% endfor %}{# package #}
