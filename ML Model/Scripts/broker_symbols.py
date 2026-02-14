# broker_symbols.py
import MetaTrader5 as mt5

class BrokerSymbols:
    """
    Handles broker-specific symbol name mapping.
    Supports multiple brokers by checking the actual account server.
    """
    
    # Define symbol mappings for different brokers, if you have a different broker, please add your broker symbols here.
    # Format: {broker_server_name: {generic_symbol: broker_symbol}}
    SYMBOL_MAPPINGS = {
        # IC Markets (default)
        'ICMarkets': {
            'us30': 'US30',
            'us2000': 'US2000', 
            'btcusd': 'BTCUSD'
        },
        # Dukascopy
        'Dukascopy': {
            'us30': 'USA30.IDX',
            'us2000': 'USSC2000.IDX',
            'btcusd': 'BTCUSD'
        },
        # Add more brokers as needed
    }
    
    @staticmethod
    def get_broker_name() -> str:
        """
        Get the broker name from the MT5 account server.
        Returns the server name for mapping lookup.
        """
        try:
            account_info = mt5.account_info()
            if account_info:
                server = account_info.server
                server_lower = server.lower()
                
                # Check for broker patterns in server name
                if 'icmarkets' in server_lower:
                    return 'ICMarkets'
                elif 'dukascopy' in server_lower:
                    return 'Dukascopy'
                else:
                    # Default to IC Markets if unknown broker
                    print(f"Unknown broker server: {server}. Using default IC Markets mapping.")
                    return 'ICMarkets'
            return 'ICMarkets'  # Default
        except Exception as e:
            print(f"Error getting broker name: {e}. Using default IC Markets mapping.")
            return 'ICMarkets'
    
    @staticmethod
    def get_broker_symbol(generic_symbol: str) -> str:
        """
        Convert generic symbol to broker-specific symbol.
        
        Args:
            generic_symbol: The generic symbol name (e.g., 'us30', 'us2000', 'btcusd')
            
        Returns:
            Broker-specific symbol name
        """
        broker_name = BrokerSymbols.get_broker_name()
        
        # Get mapping for current broker, fallback to default if not found
        broker_mapping = BrokerSymbols.SYMBOL_MAPPINGS.get(
            broker_name, 
            BrokerSymbols.SYMBOL_MAPPINGS['ICMarkets']
        )
        
        # Return broker-specific symbol or original if not mapped
        return broker_mapping.get(generic_symbol.lower(), generic_symbol)
    
    @staticmethod
    def get_all_broker_symbols(generic_symbols: list) -> dict:
        """
        Convert list of generic symbols to broker-specific symbols.
        
        Args:
            generic_symbols: List of generic symbol names
            
        Returns:
            Dictionary mapping {generic_symbol: broker_symbol}
        """
        broker_name = BrokerSymbols.get_broker_name()
        broker_mapping = BrokerSymbols.SYMBOL_MAPPINGS.get(
            broker_name, 
            BrokerSymbols.SYMBOL_MAPPINGS['ICMarkets']
        )
        
        result = {}
        for symbol in generic_symbols:
            result[symbol] = broker_mapping.get(symbol.lower(), symbol)
        
        return result
    
    @staticmethod
    def print_broker_info():
        """Print current broker information for debugging"""
        try:
            account_info = mt5.account_info()
            if account_info:
                print(f"Broker Server: {account_info.server}")
                print(f"Broker Name: {BrokerSymbols.get_broker_name()}")
                print(f"Account: {account_info.login}")
            else:
                print("No account info available")
        except Exception as e:
            print(f"Error getting broker info: {e}")