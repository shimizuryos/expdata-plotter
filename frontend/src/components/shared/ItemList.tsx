import React from "react";

interface ItemListProps<T> {
    title?: string;
    items: T[];
    renderItem: (item: T) => React.ReactNode;
    onDelete?: (item: T) => void;
    emptyMessage?: string;
}

export function ItemList<T>({ title, items, renderItem, onDelete, emptyMessage }: ItemListProps<T>) {
    if (items.length === 0) {
        return null;
        // Or return <p className="text-gray-500 text-sm">{emptyMessage}</p> if desired
    }

    return (
        <div className="bg-gray-50 p-3 rounded border border-gray-200 mt-2 max-h-40 overflow-y-auto">
            {title && <h3 className="text-xs font-bold text-gray-500 mb-2 uppercase">{title}</h3>}
            <ul className="space-y-1">
                {items.map((item, index) => (
                    <li key={index} className="flex justify-between items-center text-sm border-b border-gray-200 last:border-0 pb-1">
                        <span className="truncate mr-2">{renderItem(item)}</span>
                        {onDelete && (
                            <button
                                onClick={() => onDelete(item)}
                                className="text-red-500 hover:text-red-700 px-2 text-xs font-medium hover:bg-red-50 rounded"
                            >
                                Delete
                            </button>
                        )}
                    </li>
                ))}
            </ul>
        </div>
    );
}
